import numpy as np
import RMFsolver.constants as const 

# collection of various EOS parameters for RMFsolver.m module , collected by Alexander Haber, Oct 2022  

# For Lagrangian definition we follow https://neutronstars.utk.edu/code/o2scl/html/class/eos_had_rmf.html#eos-had-rmf  
# para=[[mn,mp],[m_sigma,m_omega,m_rho],[I3n,I3p],[gsn,gsp],[gwn,gwp],[grn,grp],[b,c,Mn_scale],[omega4_coupling],[rho4_coupling],[b1,b2,b3,a1,a2,a3,a4,a5,a6],nsat,bar,bag constant]; 
# Parameters for TMA EOS from compose,https://compose.obspm.fr/eos/28, original paper http://dx.doi.org/10.1016/0375-9474(95)00161-S  
paraTMA=[[938.9,938.9],[519.151,781.95,768.1],[-.5,.5],[-10.055,-10.055],[12.842,12.842],[2*3.8,2*3.8],[0.328*const.MeV_fm/939/(10.055)**3,38.862/10.055**4,939],[151.590*6/12.842**4],[0],[0,0,0,0,0,0,0,0,0],0.147*const.MeV_fm**3,-1,0];
# Parameters for SFHo EOS from compose, values in Steiner, Fischer et.al. are wrong!
#https://compose.obspm.fr/eos/34 
paraSFHo=[[939.565,938.272],[2.3689528914*const.MeV_fm,3.9655047020*const.MeV_fm,3.8666788766*const.MeV_fm],[-.5,.5],[2.3689528914*3.1791606374,2.3689528914*3.1791606374],[2.275188529*3.9655047020,2.275188529*3.9655047020],[2.4062374629*3.8666788766,2.4062374629*3.8666788766],[7.3536466626*10**(-3),-3.8202821956*10**(-3),939],[-1.6155896062*10**(-3)],[4.1286242877*10**(-3)],[5.5118461115,-1.8007283681*const.MeV_fm**(-2),4.2610479708*10**2*const.MeV_fm**(-4),-1.9308602647*10**(-1)*const.MeV_fm,5.6150318121*10**(-1),2.8617603774*10**(-1)*const.MeV_fm**(-1),2.7717729776*const.MeV_fm**(-2),1.2307286924*const.MeV_fm**(-3),6.1480060734*10**(-1)*const.MeV_fm**(-4)],0.1583*const.MeV_fm**3,-1,0];
# Parameters for SFHo EOS from compose, values in Steiner, Fischer et.al. are wrong!
#https://compose.obspm.fr/eos/34 
const.MeV_fm_SFHoW=197.3269631;
paraSFHoWRONG=[[939.565346,938.272013],[2.3714*const.MeV_fm_SFHoW,762.5,770],[-.5,.5],[2.3714*3.1780,2.3714*3.1780],[762.5/const.MeV_fm_SFHoW*2.2726,762.5/const.MeV_fm_SFHoW*2.2726],[770/const.MeV_fm_SFHoW*2.4047,770/const.MeV_fm_SFHoW*2.4047],[7.4653*10**(-3),-4.0887*10**(-3),939],[-1.7013*10**(-3)],[3.4525*10**(-3)],[5.8729,-1.6442*const.MeV_fm_SFHoW**(-2),3.1464*10**2*const.MeV_fm_SFHoW**(-4),-2.3016*10**(-1)*const.MeV_fm_SFHoW,5.7972*10**(-1),3.4446*10**(-1)*const.MeV_fm_SFHoW**(-1),3.4593*const.MeV_fm_SFHoW**(-2),1.3473*const.MeV_fm_SFHoW**(-3),6.6061*10**(-1)*const.MeV_fm_SFHoW**(-4)],0.1583*const.MeV_fm_SFHoW**3,-1,0];

# parameters for GM1 EOS from compose, correspods to K=300 in Glendenning https://compose.obspm.fr/eos/54
# https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.67.2414 
paraGM1=[[939,939],[550,783,770],[-.5,.5],[9.571775562457342,9.571775562457342],[10.611024696740092,10.611024696740092],[8.19657031394559,8.19657031394559],[0.002947,-0.00107,939],[0],[0],[0,0,0,0,0,0,0,0,0],0.153*const.MeV_fm**3,-1,0];
#  Bogotma-Bodmer model from Schmitt - dense matter in compact stars: model with scalar interactions  
paraGM1SYM=[[939,939],[550,783,770],[-.5,.5],[np.sqrt(4*np.pi*6.003),np.sqrt(4*np.pi*6.003)],[np.sqrt(4*np.pi*5.948),np.sqrt(4*np.pi*5.948)],[0,0],[0.00795,0.0006952,939],[0],[0],[0,0,0,0,0,0,0,0,0],0.153*const.MeV_fm**3,-1,0];
  
#IUF-EOS from https://arxiv.org/pdf/1008.3030.pdf 
#for different nomenclature see https://neutronstars.utk.edu/code/o2scl/eos/html/class/eos_had_rmf.html#eos-had-rmf
#translation: b1=Subscript[\[CapitalLambda], v]Subscript[g**2, v], all other b and a are 0   
paraIUF=[[939,939],[491.5,782.500,763.000],[-.5,.5],[np.sqrt(99.4266),np.sqrt(99.4266)],[np.sqrt(169.8349),np.sqrt(169.8349)],[np.sqrt(184.6877),np.sqrt(184.6877)],[3.3808/(2*939),0.000296/6,939],[0.03],[0],[0.046*169.8349,0,0,0,0,0,0,0,0],0.155*const.MeV_fm**3,-1,0];

# TM1e RMF from https://arxiv.org/abs/2001.10143v1 , also on compose with BPS outer crust and Thomas-Fermi calculation for inner crust:https://compose.obspm.fr/eos/221 
paraTM1e=[[938,938],[511.19777,783,770],[-.5,.5],[-10.0289,-10.0289],[12.6139,12.6139],[13.9714,13.9714],[7.2325*const.MeV_fm/938/10.0289**3,0.6183/10.0289**4,938],[6*71.3075/12.6139**4],[0],[0.0429*12.6139**2,0,0,0,0,0,0,0,0],0.145*const.MeV_fm**3,-1,0]
# QMC-RMF1/2/3/4 published in https://arxiv.org/abs/2205.10283 and on compose for T=0 \[Beta]-equil:https://compose.obspm.fr/eos/275
# https://compose.obspm.fr/eos/276
# https://compose.obspm.fr/eos/277
# https://compose.obspm.fr/eos/278 
# coupling constants and all parameters of the 4 models 
# para=[[mn,mp],[m_sigma,m_omega,m_rho],[I3n,I3p],[gsn,gsp],[gwn,gwp],[grn,grp],[b,c,Mn_scale],[omega4_coupling],[rho4_coupling],[b1,b2,b3,a1,a2,a3,a4,a5,a6],nsat,bar,bag constant]; 
paraQMCRMF1=[[939,939],[491.5,782.5,763.],[-0.5,0.5],[7.54,7.54],[8.43,8.43],[10.88,10.88],[0.0073,0.0035,939],[0],[0],[7.89,0,0,0,0,0,0,0,0],1.2283363368456012e+6,-1,-612215];
paraQMCRMF2=[[939,939],[491.5,782.5,763.],[-0.5,0.5],[7.82,7.82],[8.99,8.99],[11.24,11.24],[0.0063,-0.0009,939],[0],[0],[8.02,0,0,0,0,0,0,0,0],1.2356201822425767e+6,-1,-463438];
paraQMCRMF3=[[939,939],[491.5,782.5,763.],[-0.5,0.5],[8.32,8.32],[9.76,9.76],[11.02,11.02],[0.0063,-0.006,939],[0],[0],[5.87,0,0,0,0,0,0,0,0],1.2053508748737855e+6,-1,-707480];
paraQMCRMF4=[[939,939],[491.5,782.5,763.],[-0.5,0.5],[8.21,8.21],[9.94,9.94],[12.18,12.18],[0.0041,-0.0021,939],[0],[0],[10.43,0,0,0,0,0,0,0,0],1.245066243773348e+6,-1,-206742];
paraQMCRMFx=[[1,paraQMCRMF1],[2,paraQMCRMF2],[3,paraQMCRMF3],[4,paraQMCRMF4]];

paraQMCRMF1_3D=[[939,939],[491.5,782.5,763.],[-0.5,0.5],[7.54,7.54],[8.43,8.43],[10.88,10.88],[0.0073,0.0035,939],[0],[0],[7.89,0,0,0,0,0,0,0,0],1.2283363368456012e+6,-1,-577307];
paraQMCRMF2_3D=[[939,939],[491.5,782.5,763.],[-0.5,0.5],[7.82,7.82],[8.99,8.99],[11.24,11.24],[0.0063,-0.0009,939],[0],[0],[8.02,0,0,0,0,0,0,0,0],1.2356201822425767e+6,-1,-460210];
paraQMCRMF3_3D=[[939,939],[491.5,782.5,763.],[-0.5,0.5],[8.32,8.32],[9.76,9.76],[11.02,11.02],[0.0063,-0.006,939],[0],[0],[5.87,0,0,0,0,0,0,0,0],1.2053508748737855e+6,-1,-723934];
paraQMCRMF4_3D=[[939,939],[491.5,782.5,763.],[-0.5,0.5],[8.21,8.21],[9.94,9.94],[12.18,12.18],[0.0041,-0.0021,939],[0],[0],[10.43,0,0,0,0,0,0,0,0],1.245066243773348e+6,-1,-150000];
paraQMCRMFx_3D=[[1,paraQMCRMF1_3D],[2,paraQMCRMF2_3D],[3,paraQMCRMF3_3D],[4,paraQMCRMF4_3D]];

