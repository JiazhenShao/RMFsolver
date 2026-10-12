"""Analytical contour roots with pointwise persistence and shared speed brackets."""
import copy
from pathlib import Path
import threading
import time
import numpy as np
from _logT_common import CONTROLS, atomic_save, load, target, temperature_id, point, next_density


class AnalyticRun:
    def __init__(self, directory, inputs, identity, *, controls=None, evaluator=None,
                 evaluation_saved=None, target_saved=None, boundary_saved=None):
        self.directory=Path(directory)
        self.output=self.directory/'isothermal-analytic-logT.npy'
        self.inputs=inputs
        self.controls=dict(CONTROLS,**(controls or {}))
        self.lock=threading.RLock()
        self.evaluator=evaluator
        self.evaluation_saved=evaluation_saved or (lambda e:None)
        self.target_saved=target_saved or (lambda t:None)
        self.boundary_saved=boundary_saved or (lambda b:None)
        self.payload=dict(schema=2,method='analytic_isothermal',identity=identity,
            controls=self.controls,domain_fingerprint=inputs['fingerprint'],
            boundaries=copy.deepcopy(inputs['boundary_cache']),targets={},evaluations=[],points={},
            work_complete=False,contours_complete=False)
        if self.output.exists():
            self.payload=load(self.output)
            if self.payload['identity']!=identity or self.payload['controls']!=self.controls:
                raise ValueError('Checkpoint identity/controls incompatible; use a separate output directory')
            for e in self.payload['evaluations']:
                if e['status']=='running':
                    e.update(status='interrupted',runtime_s=e['reserved_seconds'])
            for row in self.payload['targets'].values():
                if row['status']=='running':
                    row['status']='pending'
            for row in self.payload['boundaries'].values():
                if row['status']=='running':
                    row['status']='interrupted'
        self.save()

    def save(self):
        with self.lock:
            self.payload['updated_unix_s']=time.time()
            atomic_save(self.output,self.payload)

    def prepare_boundary(self,T):
        key=temperature_id(T)
        with self.lock:
            prior=self.payload['boundaries'].get(key)
            if prior and prior['status']!='interrupted':
                return prior
            if prior and prior.get('attempts',1)>=2:
                return prior
            seeds={}
            for name,field in (('lower','inner_boundary'),('upper','outer_boundary')):
                curve=np.asarray(self.inputs['boundary_geometry'][field],float)
                curve=curve[np.argsort(curve[:,1])]
                seeds[name]=float(np.interp(T,curve[:,1],curve[:,0]))
            attempts=(prior.get('attempts',1) if prior else 0)+1
            self.payload['boundaries'][key]=dict(status='running',T_0minus=T,attempts=attempts)
            self.save()
        from _logT_trial import bounded_call
        result=bounded_call('boundary',timeout=self.controls['boundary_timeout'],T=T,seeds=seeds,n0=self.inputs['n0'])
        with self.lock:
            row=dict(result,T_0minus=T,attempts=attempts)
            self.payload['boundaries'][key]=row
            self.save()
            self.boundary_saved(copy.deepcopy(row))
        return row

    def spent(self,task):
        return sum(e.get('runtime_s',e['reserved_seconds']) for e in self.payload['evaluations']
                   if e['target_id']==task['id'])

    def shoot(self,task,physical):
        with self.lock:
            cached=self.payload['points'].get(physical['id'])
            if cached:
                return cached
            events=self.payload['evaluations']
            prior=[e for e in events if e['target_id']==task['id']]
            remaining=self.controls['target_timeout']-self.spent(task)
            if len(prior)>=self.controls['evaluations_per_target'] or remaining<=0:
                return None
            if any(e['point']['id']==physical['id'] for e in events):
                return None
            event=dict(id=len(events),target_id=task['id'],point=physical,status='running',
                reserved_seconds=min(self.controls['trial_timeout'],remaining))
            events.append(event)
            self.save()  # Persist reservation before launching a worker.
        started=time.monotonic()
        try:
            if self.evaluator is not None:
                result=self.evaluator(physical)
            else:
                from _logT_trial import bounded_call
                result=bounded_call('analytical',timeout=event['reserved_seconds'],point=physical)
        except Exception as exc:
            result=dict(success=False,status='exception',message=f'{type(exc).__name__}: {exc}')
        with self.lock:
            valid=bool(result.get('validated') and np.isfinite(result.get('velocity_m_s',np.nan))
                       and result['velocity_m_s']>0)
            classified_zero=bool(result.get('success') and result.get('status') in
                                 ('stable_neutron_matter','isothermal_coexistence'))
            accepted=None
            if valid or classified_zero:
                accepted=dict(physical,velocity_m_s=float(result['velocity_m_s']),result=result,
                              source='analytical_xi_minus0p85')
                self.payload['points'][physical['id']]=accepted
            event.update(result=result,status='success' if valid else result.get('status','failed'),
                         runtime_s=time.monotonic()-started)
            self.save()
            self.evaluation_saved(copy.deepcopy(event))
            return accepted

    def finish_target(self,task,status,root=None,message=None):
        with self.lock:
            task.update(status=status,root=root)
            if message:
                task['message']=message
            self.save()
            self.target_saved(copy.deepcopy(task))

    def solve_temperature(self,T,levels):
        boundary=self.prepare_boundary(T)
        for level in sorted(levels):
            task_id=target(level,T)['id']
            with self.lock:
                task=self.payload['targets'].setdefault(task_id,target(level,T))
                if task['status'] not in ('pending','running'):
                    continue
                task['status']='running'
                self.save()
            if boundary['status']=='outside_window':
                self.finish_target(task,'outside_window',message=boundary.get('reason','No admissible density interval'))
                continue
            if boundary['status']!='success':
                self.finish_target(task,'unresolved',message='No validated static boundary interval')
                continue
            lo,hi=boundary['lower'],boundary['upper']
            with self.lock:
                samples=[p for p in self.payload['points'].values() if p['T_0minus']==T]
                failures=[e['point']['nB_0minus_over_n0'] for e in self.payload['evaluations']
                    if e['point']['T_0minus']==T and e['status']!='success'
                    and e['point']['id'] not in self.payload['points']]
            if not boundary.get('lower_clipped',False):
                samples.append(dict(nB_0minus_over_n0=lo,velocity_m_s=0.,source='coexistence_limit'))
            else:
                edge=self.shoot(task,point(T,lo,self.inputs['n0']))
                if edge is None:
                    self.finish_target(task,'unresolved',message='Could not validate speed at density-window edge')
                    continue
                if edge['velocity_m_s']>level*(1+self.controls['speed_rtol']):
                    self.finish_target(task,'outside_window',message='Target below analytical speed at density-window edge')
                    continue
                samples.append(edge)
            if boundary.get('upper_clipped',False):
                edge=self.shoot(task,point(T,hi,self.inputs['n0']))
                if edge is None:
                    self.finish_target(task,'unresolved',message='Could not validate speed at upper density-window edge')
                    continue
                if edge['velocity_m_s']<level*(1-self.controls['speed_rtol']):
                    self.finish_target(task,'outside_window',message='Target above analytical speed at upper density-window edge')
                    continue
                samples.append(edge)
            predicted=lo+(hi-lo)*np.clip(level/1.e5,1.e-4,.1)
            found=None
            for _ in range(self.controls['evaluations_per_target']+1):
                found=next((p for p in samples if p.get('result',{}).get('validated')
                            and abs(p['velocity_m_s']/level-1)<=self.controls['speed_rtol']),None)
                if found or self.spent(task)>=self.controls['target_timeout']:
                    break
                density=next_density(samples,failures,predicted,level,(lo,hi))
                if density is None:
                    break
                answer=self.shoot(task,point(T,density,self.inputs['n0']))
                if answer:
                    samples.append(answer)
                else:
                    failures.append(density)
                    # No velocities are assigned to failed or saturated trials.
                    last=next((e for e in reversed(self.payload['evaluations'])
                               if e['point']['id']==point(T,density,self.inputs['n0'])['id']),None)
                    if last and last.get('result',{}).get('status')=='composition_ceiling_saturated':
                        hi=min(hi,density)
            self.finish_target(task,'success' if found else 'unresolved',root=found,
                message=None if found else 'No accepted analytical root within point/time allowance')

    def finalize(self):
        with self.lock:
            statuses=[t['status'] for t in self.payload['targets'].values()]
            self.payload['work_complete']=bool(statuses) and all(s not in ('pending','running') for s in statuses)
            self.payload['contours_complete']=bool(statuses) and all(s in ('success','outside_window') for s in statuses)
            self.save()
