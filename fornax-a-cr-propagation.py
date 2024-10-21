# --------------------------------------------------------------------
# --------------------------------------------------------------------

from crpropa import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

# --------------------------------------------------------------------
# --------------------- SIMULATION FUNCTIONS ------------------------- 
# --------------------------------------------------------------------
    
def simulate(distance, min_energy, max_energy, spectral_index, injection, n_events):
    EBL = IRB_Gilmore12()
    neutrinos = True
    photons = True
    electrons = True
    redshift_dependance = True

    # source
    redshift = comovingDistance2Redshift(distance)
    source = Source()
    source.add(SourcePosition(Vector3d(distance, 0, 0)))
    source.add(SourceDirection(Vector3d(-1, 0, 0)))
    source.add(SourceRedshift1D())
    source.add(SourcePowerLawSpectrum(min_energy, max_energy, spectral_index))
    composition = SourceMultipleParticleTypes()
    for element in range(len(injection)):
        composition.add(injection[element])
    source.add(composition)

    # interactions
    sim = ModuleList()
    sim.add(Redshift()) # adiabatic energy loss
    sim.add(PhotoPionProduction(CMB()))
    sim.add(PhotoPionProduction(EBL))
    sim.add(ElectronPairProduction(CMB()))
    sim.add(ElectronPairProduction(EBL))
    sim.add(PhotoDisintegration(CMB()))
    sim.add(PhotoDisintegration(EBL))
    sim.add(NuclearDecay())

    # observer
    arquivo = "SIM_DATA.txt"
    output = TextOutput(arquivo, Output.Event1D)
    output.setEnergyScale(eV)
    output.setLengthScale(Mpc)
    output.enable(output.CreatedPositionColumn)
    observer = Observer()
    observer.add(Observer1D())
    #observer.add(ObserverElectronVeto())
    observer.onDetection(output)
    
    # assemble
    breakEnergy = MinimumEnergy(1 * EeV)
    sim.add(SimplePropagation(0.1 * kpc, 1 * Mpc)) # min_step, max_step
    sim.add(breakEnergy)
    sim.add(observer)
    sim.setShowProgress(True)
    sim.run(source, events, True)
    output.close()
    
    return output

# ---------------------------------------------------------------------
# ------------------- DATA MANIPULATION FUNCTIONS ---------------------
# ---------------------------------------------------------------------

def read_simulation(file):
    # read the sim and return a pandas dataframe
    with open(file, 'r') as f:
        line = f.readline()
        line = line.replace('\n', '').replace('#', '')
        names = line.split('\t')[1:]
    df = pd.read_csv(file, delimiter = '\t',comment = '#', names = names)
    return df

def reweight_simulation(df, alpha, Rmax, column_name='W1', alpha0=1.):
    '''
    alpha: desired spectral index
    -alpha0: spectral index used in the simulation
    Rmax: spectrum cuts off above Emax = Z * Rmax (Z: atomic num)
    '''
    '''
    composition_factors = { # c# (abundância do sistema solar)
        1000010010: 0.922,
        1000020040: 0.078,
        1000070140: 0.0008,
        1000140280: 0.00008,
        1000260560: 0.00003
    }
    
    composition_factors = { # c1 (população estelar nos lóbulos de cenA)
        1000010010: 0.916,
        1000020040: 0.083,
        1000070140: 0.00042,
        1000140280: 0.000057,
        1000260560: 0.000015
    }
    
    composition_factors = { # c2 (enriquecimento da composição solar por fermi)
        1000010010: 0.849,
        1000020040: 0.1437,
        1000070140: 0.0052,
        1000140280: 0.001,
        1000260560: 0.00072
    }
    '''
    composition_factors = { # c4 (ajuste dos dados do PAO)
        1000010010: 0.7692,
        1000020040: 0.1538,
        1000070140: 0.0461,
        1000140280: 0.0231,
        1000260560: 0.00759
    }
    '''
    composition_factors = { # c3 (estrelas wolf-rayet)
        1000010010: 0,
        1000020040: 0.62,
        1000070140: 0.37,
        1000140280: 0.01,
        1000260560: 0
    }
    '''
    compute_weight = lambda e0, z0, composition: (e0 ** (alpha0-alpha)) * np.exp(- e0 / (z0 * Rmax)) * composition  
    weight = np.array([ # calcula os pesos usando o fator associado ao 'ID0' do dicionário
        compute_weight(df['E0'][i], chargeNumber(int(df['ID0'][i])), composition_factors.get(int(df['ID0'][i]), 0))
        for i in range(len(df['E0']))
    ])
    df[column_name] = weight

def compute_spectra(df, column_name = 'W1'):
    bins = np.logspace(15, 25, 100) # returns numbers spaced evenly on a log scale; min, max, step
    y, edges = np.histogram(df['E'], bins = bins, weights = df[column_name]) # histogram of weighted energies
    #x = edges[:-1] + ((edges[1:] - edges[:-1]) - 2.) # adjusts the bin centers
    x = (edges[:-1] + edges[1:]) / 2
    y /= np.diff(edges) # divides the number of events y by the bin size so we get the energy density per interval 
    #y *= (x**2) # for E^2 dN/dE
    y *= (x**3)  # for E^3 dN/dE
    return x, y

# --------------------------------------------------------------------
# --------------------- RUNNING THE SIMULATION -----------------------
# --------------------------------------------------------------------

min_energy = 10**(18) * eV
max_energy = 10**(21) * eV
spectral_index = -1
events = 2000000

alpha = 2
Rmax = 1.45e18 # volts

distance = 20.8 * Mpc
H = nucleusId(1, 1)
He = nucleusId(4, 2)        # (4, 2)
N = nucleusId(14, 7)        # (17, 7)
Si = nucleusId(28, 14)      # (28, 14)
Fe = nucleusId(56, 26)      # (56, 26)

injection = [H, He, N, Si, Fe]
names = ["H", "He", "N", "Si", "Fe"]
'''
output = simulate(distance, min_energy, max_energy, spectral_index, injection, events)
output.close()
'''
data = read_simulation("SIM_DATA.txt")
reweight_simulation(data, alpha, Rmax)

# --------------------------------------------------------------------
# ------------------------- FILTERING DATA ---------------------------
# --------------------------------------------------------------------

'''
# checagem dos valores diferentes de ID da simulação (diferentes partículas detectadas)
unique_values = data['ID'].unique().tolist()
print(unique_values)
'''
id_h = [1000010010]
ids_2 = [1000020040, 1000020030, 1000010020, 1000010030]
ids_5 = [1000060130, 1000070140, 1000060120, 1000040090, 1000080170, 1000070150, 1000050110, 1000100200, 1000050100, 1000080160, 1000090190, 1000100210, 1000080180, 1000060140, 1000090180, 1000030060, 1000040070, 1000040100, 1000060110]
ids_22 = [1000120260, 1000140280, 1000120240, 1000130270, 1000140290, 1000180370, 1000110230, 1000120250, 1000100220, 1000130260, 1000160350, 1000160340, 1000110220, 1000140300, 1000170360, 1000160360, 1000170350, 1000160330, 1000140310, 1000160320, 1000170370, 1000150330, 1000180360, 1000140320]
ids_38 = [1000260560, 1000260550, 1000200440, 1000210450, 1000250530, 1000260540, 1000250540, 1000250550, 1000180380, 1000200430, 1000220470, 1000240520, 1000240510, 1000220460, 1000190390, 1000250520, 1000150320, 1000150310, 1000240500, 1000230480, 1000220480, 1000230490, 1000200420, 1000240530, 1000190400, 1000230500, 1000200410, 1000190410, 1000190420, 1000210460, 1000230510, 1000240540, 1000200400, 1000210470, 1000180390, 1000220490, 1000190430, 1000180400, 1000210430, 1000260520, 1000170380, 1000220500, 1000240480, 1000240490, 1000210480]
all_ids = np.concatenate((id_h, ids_2, ids_5, ids_22, ids_38))

H_x, H_y = compute_spectra(data[data['ID'].isin(id_h)])
He_x, He_y = compute_spectra(data[data['ID'].isin(ids_2)]) #A:2-4
N_x, N_y = compute_spectra(data[data['ID'].isin(ids_5)]) #A:5-21
Si_x, Si_y = compute_spectra(data[data['ID'].isin(ids_22)]) #A:22-37
Fe_x, Fe_y = compute_spectra(data[data['ID'].isin(ids_38)]) #A:38-56
all_x, all_y = compute_spectra(data[data['ID'].isin(all_ids)])

# --------------------------------------------------------------------
# --------------------------------------------------------------------

import scipy.stats
from scipy import stats
import os.path
from zipfile import ZipFile

def AugerLoad(fdir, file):
    for loc in ['.', '..', 'augeropendata']:
        fname = os.path.join(loc, fdir, file)
        if os.path.isfile(fname):
            return open(fname)
        zname = os.path.join(loc, fdir+".zip")
        if os.path.isfile(zname):
            with ZipFile(zname) as myzip:
                return myzip.open(os.path.join(fdir, file))

data = pd.read_csv(AugerLoad("summary", "dataSummarySD1500.csv"))

data = data[data["sd_exposure"]>0]
data = data[(data.sd_energy.notna()) & (data.sd_energy>2.5)]
data = data.sort_values(by="sdid")
exposure = data["sd_exposure"].iat[-1]
energy = data.drop_duplicates("id")["sd_energy"]

log_E_min = 0.4
E_bins = 20
E_bin_size = 0.1
log_E_max = log_E_min + E_bins * E_bin_size

log_bins = np.linspace(log_E_min, log_E_max, E_bins + 1)
log_bin_centers = log_bins[:-1] + 0.05
bins = pow(10, log_bins);
bin_energy = pow(10, log_bin_centers)
bin_width = bins[1:] - bins[:-1]

h = np.histogram(energy, bins)[0]

alpha = 0.16
beta = 0.16
lim_low = (h - np.nan_to_num(0.5 * scipy.stats.chi2.ppf(alpha, 2 * h)) )
lim_up = ( 0.5 * scipy.stats.chi2.ppf(1 - beta, 2 * (h + 1)) - h)

cut_nz = h > 0
cut_z = h == 0

normalization = exposure * bin_width * 1e18
flux = h[cut_nz] / normalization[cut_nz]
flux_lower = lim_low[cut_nz] / normalization[cut_nz]
flux_upper = lim_up[cut_nz] / normalization[cut_nz]

bin_energy18 = bin_energy * 1e18
bin_energy18_3 = bin_energy18**3
flux_E3 = flux * bin_energy18_3[cut_nz]
flux_E3_lower = flux_lower * bin_energy18_3[cut_nz]
flux_E3_upper = flux_upper * bin_energy18_3[cut_nz]

FC_90CL_0 = 2.44

FC_CL    = FC_90CL_0 / normalization[cut_z]
FC_CL_E3 = FC_CL * bin_energy18_3[cut_z]

FC_CLt    = FC_90CL_0 / normalization[cut_z]
FC_CL_E3t = FC_CLt * bin_energy18_3[cut_z]

# --------------------------------------------------------------------
# ------------------------------ PLOT --------------------------------
# --------------------------------------------------------------------

limite_inferior = 10**18.3
limite_superior = 10**18.5

# filtragem dos pontos de referência (bin_energy18[cut_nz], flux_E3) para o intervalo desejado
mask_ref = (bin_energy18[cut_nz] >= limite_inferior) & (bin_energy18[cut_nz] <= limite_superior)
x_ref_filtered = bin_energy18[cut_nz][mask_ref]
y_ref_filtered = flux_E3[mask_ref]

# filtragem dos pontos de all_x e all_y para o intervalo desejado
mask_all = (all_x >= limite_inferior) & (all_x <= limite_superior)
x_all_filtered = all_x[mask_all]
y_all_filtered = all_y[mask_all]

# verificação se os pontos filtrados correspondem corretamente em quantidade
if len(x_all_filtered) != len(x_ref_filtered):
    # print("Atenção: o número de pontos filtrados em 'all_x' e 'bin_energy18[cut_nz]' não coincide.")
    # caso necessário, realizar interpolação para garantir correspondência ponto a ponto
    y_all_filtered_interp = np.interp(x_ref_filtered, all_x, all_y)
else:
    y_all_filtered_interp = y_all_filtered

# cálculo do fator de escala alpha usando os pontos filtrados no intervalo desejado
alpha_filtered = np.dot(y_ref_filtered, y_all_filtered_interp) / np.dot(y_all_filtered_interp, y_all_filtered_interp)

# --------------------------------------------------------------------
# --------------------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.title('Espectro de energia de Fornax A (C4)', fontsize=15, fontname='DejaVu Serif')

plt.errorbar(bin_energy18[cut_nz], flux_E3, yerr=[flux_E3_lower, flux_E3_upper], fmt="o", markersize=5, label='Pierre Auger', c='deeppink')

plt.loglog(Fe_x, Fe_y*alpha_filtered, linewidth=1, ls='-', marker='o', markersize=3, c='#ae4c7f', label='A $\geq$ 38')
plt.loglog(Si_x, Si_y*alpha_filtered, linewidth=1, ls='-', marker='o', markersize=3, c='#E8AF02', label='23 $\leq$ A $\leq$ 37')
plt.loglog(N_x, N_y*alpha_filtered, linewidth=1, ls='-', marker='o', markersize=3, c='#6bd58d', label='5 $\leq$ A $\leq$ 22')
plt.loglog(He_x, He_y*alpha_filtered, linewidth=1, ls='-', marker='o', markersize=3, c='#ed502e', label='2 $\leq$ A $\leq$ 4')
plt.loglog(H_x, H_y*alpha_filtered, linewidth=1, ls='-', marker='o', markersize=3, c='#4e7aff', label='A = 1')
plt.loglog(all_x, all_y*alpha_filtered, linewidth=2, ls='-', marker='o', markersize=3, c='black', label='Total')

plt.xscale("log")
plt.yscale("log")
plt.xlim(10**(18.5), 1e21)
plt.ylim(1e34, 1e39)

plt.ylabel(r'E$^{3}$ dN/dE [km$^{-2}$ sr$^{-1}$ yr$^{-1}$ eV$^{2}$]', fontname='DejaVu Serif', fontsize=13)
plt.xlabel('$E$ [eV]', fontname='DejaVu Serif', fontsize=13)
plt.legend(loc='upper right')
#plt.grid()

plt.savefig('espectro.png', dpi=300)
plt.show()
