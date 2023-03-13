from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np

from photutils import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.aperture import ApertureStats

import random


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Lectura archivos .fits de filtro F606W
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

hdul_F606W = fits.open(
    '/home/javier/Escritorio/TTB/nircam1/egs_all_acs_wfc_f606w_030mas_v1.9_nircam1_mef.fits')
hdr_F606W = hdul_F606W[0].header
data_F606W = hdul_F606W[1].data

pixel_scale = 0.03  # arcsecs/pixel


"""""""""""""""""""""""""""""""""""""""""""""
Generación de apertura y anillos de fondo
"""""""""""""""""""""""""""""""""""""""""""""

# CEERSYJ-0012159472

aperture_diameter_arcsecs_472 = 0.3  # 0.35
aperture_radius_472 = aperture_diameter_arcsecs_472 / pixel_scale / 2

xc_472_F606W, yc_472_F606W = 2563.17, 2677.76
position_472_F606W = (xc_472_F606W, yc_472_F606W)

central_aperture_F606W_472 = CircularAperture(
    position_472_F606W, r=aperture_radius_472)
annulus_aperture_F606W_472 = CircularAnnulus(
    position_472_F606W, r_in=aperture_radius_472*2, r_out=aperture_radius_472*3)

delta_arcsecs_472_F606W = 3
delta_pix_472_F606W = delta_arcsecs_472_F606W / pixel_scale
x_min_472_F606W = xc_472_F606W - delta_pix_472_F606W
x_max_472_F606W = xc_472_F606W + delta_pix_472_F606W
y_min_472_F606W = yc_472_F606W - delta_pix_472_F606W
y_max_472_F606W = yc_472_F606W + delta_pix_472_F606W

# CEERSYJ-9586559217

aperture_diameter_arcsecs_217 = 0.3  # 0.208
aperture_radius_217 = aperture_diameter_arcsecs_217 / pixel_scale / 2

xc_217_F606W, yc_217_F606W = 3708.89, 2710.52
position_217_F606W = (xc_217_F606W, yc_217_F606W)

central_aperture_F606W_217 = CircularAperture(
    position_217_F606W, r=aperture_radius_217)
annulus_aperture_F606W_217 = CircularAnnulus(
    position_217_F606W, r_in=aperture_radius_217*2, r_out=aperture_radius_217*3)

delta_arcsecs_217_F606W = 3
delta_pix_217_F606W = delta_arcsecs_217_F606W / pixel_scale
x_min_217_F606W = xc_217_F606W - delta_pix_217_F606W
x_max_217_F606W = xc_217_F606W + delta_pix_217_F606W
y_min_217_F606W = yc_217_F606W - delta_pix_217_F606W
y_max_217_F606W = yc_217_F606W + delta_pix_217_F606W

# CEERSYJ-0012959481

aperture_diameter_arcsecs_481 = 0.3  # 0.336
aperture_radius_481 = aperture_diameter_arcsecs_481 / pixel_scale / 2

xc_481_F606W, yc_481_F606W = 2526.86, 2677.60
position_481_F606W = (xc_481_F606W, yc_481_F606W)

central_aperture_F606W_481 = CircularAperture(
    position_481_F606W, r=aperture_radius_481)
annulus_aperture_F606W_481 = CircularAnnulus(
    position_481_F606W, r_in=aperture_radius_481*2, r_out=aperture_radius_481*3)

delta_arcsecs_481_F606W = 3
delta_pix_481_F606W = delta_arcsecs_481_F606W / pixel_scale
x_min_481_F606W = xc_481_F606W - delta_pix_481_F606W
x_max_481_F606W = xc_481_F606W + delta_pix_481_F606W
y_min_481_F606W = yc_481_F606W - delta_pix_481_F606W
y_max_481_F606W = yc_481_F606W + delta_pix_481_F606W


"""""""""""""""""""""""""""""""""""""""
Gráfico de aperturas a las 3 galaxias
"""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F606W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F606W, interval=ZScaleInterval()))
central_aperture_F606W_472.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F606W_472.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_472_F606W, x_max_472_F606W, y_min_472_F606W, y_max_472_F606W])

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F606W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F606W, interval=ZScaleInterval()))
central_aperture_F606W_217.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F606W_217.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_217_F606W, x_max_217_F606W, y_min_217_F606W, y_max_217_F606W])

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F606W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F606W, interval=ZScaleInterval()))
central_aperture_F606W_481.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F606W_481.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_481_F606W, x_max_481_F606W, y_min_481_F606W, y_max_481_F606W])

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F606W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F606W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Fotometria de apertura con sustracción de fondo a CEERSYJ-0012159472,
CEERSYJ-9586559217 y CEERSYJ-0012959481 con una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

positions = [position_472_F606W, position_217_F606W, position_481_F606W]

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2

aperture = CircularAperture(positions, r=aperture_radius)
aperture_area = aperture.area

annulus_aperture = CircularAnnulus(
    positions, r_in=aperture_radius*2, r_out=aperture_radius*3)

aperstats = ApertureStats(data_F606W, annulus_aperture)
bkg_mean = aperstats.mean

total_bkg = bkg_mean * aperture_area

phot_table_F606W = aperture_photometry(data_F606W, aperture)
phot_bkgsub = phot_table_F606W['aperture_sum'] - total_bkg
phot_table_F606W['total_bkg'] = total_bkg
phot_table_F606W['aperture_sum_bkgsub'] = phot_bkgsub

for col in phot_table_F606W.colnames:
    phot_table_F606W[col].info.format = '%.8g'  # for consistent table output

print(phot_table_F606W)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Obtención de errores de la fotometría de apertura con
sustracción de fondo y para una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2
N_apertures = 50

# CEERSYJ-0012159472

cutout_472_F606W = data_F606W[int(np.floor(x_min_472_F606W)):int(np.ceil(
    x_max_472_F606W)), int(np.floor(y_min_472_F606W)):int(np.ceil(y_max_472_F606W))]

lower_x_472_F606W, upper_x_472_F606W = int(np.ceil(
    x_min_472_F606W + aperture_radius*3)), int(np.floor(x_max_472_F606W - aperture_radius*3))
lower_y_472_F606W, upper_y_472_F606W = int(np.ceil(
    y_min_472_F606W + aperture_radius*3)), int(np.floor(y_max_472_F606W - aperture_radius*3))

x_centers_472_F606W = [random.randrange(
    start=lower_x_472_F606W, stop=upper_x_472_F606W) for i in range(N_apertures)]
y_centers_472_F606W = [random.randrange(
    start=lower_y_472_F606W, stop=upper_y_472_F606W) for i in range(N_apertures)]

random_apertures_472_F606W = CircularAperture(
    zip(x_centers_472_F606W, y_centers_472_F606W), r=aperture_radius)
random_apertures_472_F606W_area = aperture.area

random_annulus_472_F606W = CircularAnnulus(zip(
    x_centers_472_F606W, y_centers_472_F606W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_472_F606W = ApertureStats(data_F606W, random_annulus_472_F606W)
annulus_bkg_mean_472_F606W = apertures_stats_472_F606W.mean

total_random_bkg_472_F606W = annulus_bkg_mean_472_F606W * \
    random_apertures_472_F606W_area

phot_table_472_F606W = aperture_photometry(
    data_F606W, random_apertures_472_F606W)
phot_bkgsub_472_F606W = phot_table_472_F606W['aperture_sum'] - \
    total_random_bkg_472_F606W
phot_table_472_F606W['total_bkg'] = total_random_bkg_472_F606W
phot_table_472_F606W['aperture_sum_bkgsub'] = phot_bkgsub_472_F606W

for col in phot_table_472_F606W.colnames:
    # for consistent table output
    phot_table_472_F606W[col].info.format = '%.8g'

print(phot_table_472_F606W)

fluxes_472_F606W = phot_table_472_F606W['aperture_sum_bkgsub']
print(len(fluxes_472_F606W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_472_F606W)))
fluxes_472_F606W = fluxes_472_F606W[np.where(
    fluxes_472_F606W < 1*np.std(fluxes_472_F606W))]
print(len(fluxes_472_F606W))
fluxes_472_F606W_std = np.std(fluxes_472_F606W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_472_F606W_std))

# CEERSYJ-9586559217

cutout_217_F606W = data_F606W[int(np.floor(x_min_217_F606W)):int(np.ceil(
    x_max_217_F606W)), int(np.floor(y_min_217_F606W)):int(np.ceil(y_max_217_F606W))]

lower_x_217_F606W, upper_x_217_F606W = int(np.ceil(
    x_min_217_F606W + aperture_radius*3)), int(np.floor(x_max_217_F606W - aperture_radius*3))
lower_y_217_F606W, upper_y_217_F606W = int(np.ceil(
    y_min_217_F606W + aperture_radius*3)), int(np.floor(y_max_217_F606W - aperture_radius*3))

x_centers_217_F606W = [random.randrange(
    start=lower_x_217_F606W, stop=upper_x_217_F606W) for i in range(N_apertures)]
y_centers_217_F606W = [random.randrange(
    start=lower_y_217_F606W, stop=upper_y_217_F606W) for i in range(N_apertures)]

random_apertures_217_F606W = CircularAperture(
    zip(x_centers_217_F606W, y_centers_217_F606W), r=aperture_radius)
random_apertures_217_F606W_area = aperture.area

random_annulus_217_F606W = CircularAnnulus(zip(
    x_centers_217_F606W, y_centers_217_F606W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_217_F606W = ApertureStats(data_F606W, random_annulus_217_F606W)
annulus_bkg_mean_217_F606W = apertures_stats_217_F606W.mean

total_random_bkg_217_F606W = annulus_bkg_mean_217_F606W * \
    random_apertures_217_F606W_area

phot_table_217_F606W = aperture_photometry(
    data_F606W, random_apertures_217_F606W)
phot_bkgsub_217_F606W = phot_table_217_F606W['aperture_sum'] - \
    total_random_bkg_217_F606W
phot_table_217_F606W['total_bkg'] = total_random_bkg_217_F606W
phot_table_217_F606W['aperture_sum_bkgsub'] = phot_bkgsub_217_F606W

for col in phot_table_217_F606W.colnames:
    # for consistent table output
    phot_table_217_F606W[col].info.format = '%.8g'

print(phot_table_217_F606W)

fluxes_217_F606W = phot_table_217_F606W['aperture_sum_bkgsub']
print(len(fluxes_217_F606W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_217_F606W)))
fluxes_217_F606W = fluxes_217_F606W[np.where(
    fluxes_217_F606W < 1*np.std(fluxes_217_F606W))]
print(len(fluxes_217_F606W))
fluxes_217_F606W_std = np.std(fluxes_217_F606W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_217_F606W_std))

# CEERSYJ-0012959481

cutout_481_F606W = data_F606W[int(np.floor(x_min_481_F606W)):int(np.ceil(
    x_max_481_F606W)), int(np.floor(y_min_481_F606W)):int(np.ceil(y_max_481_F606W))]

lower_x_481_F606W, upper_x_481_F606W = int(np.ceil(
    x_min_481_F606W + aperture_radius*3)), int(np.floor(x_max_481_F606W - aperture_radius*3))
lower_y_481_F606W, upper_y_481_F606W = int(np.ceil(
    y_min_481_F606W + aperture_radius*3)), int(np.floor(y_max_481_F606W - aperture_radius*3))

x_centers_481_F606W = [random.randrange(
    start=lower_x_481_F606W, stop=upper_x_481_F606W) for i in range(N_apertures)]
y_centers_481_F606W = [random.randrange(
    start=lower_y_481_F606W, stop=upper_y_481_F606W) for i in range(N_apertures)]

random_apertures_481_F606W = CircularAperture(
    zip(x_centers_481_F606W, y_centers_481_F606W), r=aperture_radius)
random_apertures_481_F606W_area = aperture.area

random_annulus_481_F606W = CircularAnnulus(zip(
    x_centers_481_F606W, y_centers_481_F606W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_481_F606W = ApertureStats(data_F606W, random_annulus_481_F606W)
annulus_bkg_mean_481_F606W = apertures_stats_481_F606W.mean

total_random_bkg_481_F606W = annulus_bkg_mean_481_F606W * \
    random_apertures_481_F606W_area

phot_table_481_F606W = aperture_photometry(
    data_F606W, random_apertures_481_F606W)
phot_bkgsub_481_F606W = phot_table_481_F606W['aperture_sum'] - \
    total_random_bkg_481_F606W
phot_table_481_F606W['total_bkg'] = total_random_bkg_481_F606W
phot_table_481_F606W['aperture_sum_bkgsub'] = phot_bkgsub_481_F606W

for col in phot_table_481_F606W.colnames:
    # for consistent table output
    phot_table_481_F606W[col].info.format = '%.8g'

print(phot_table_481_F606W)

fluxes_481_F606W = phot_table_481_F606W['aperture_sum_bkgsub']
print(len(fluxes_481_F606W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_481_F606W)))
fluxes_481_F606W = fluxes_481_F606W[np.where(
    fluxes_481_F606W < 1*np.std(fluxes_481_F606W))]
print(len(fluxes_481_F606W))
fluxes_481_F606W_std = np.std(fluxes_481_F606W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_481_F606W_std))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Gráfico de apertura de puntos azarosos cercanos a
CEERSYJ-0012159472, CEERSYJ-9586559217 y CEERSYJ-0012959481
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F606W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F606W, interval=ZScaleInterval()))
central_aperture_F606W_472.plot(color='yellow', lw=2)
random_apertures_472_F606W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlim((int(np.floor(x_min_472_F606W)), int(np.ceil(x_max_472_F606W))))
plt.ylim((int(np.floor(y_min_472_F606W)), int(np.ceil(y_max_472_F606W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F606W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F606W, interval=ZScaleInterval()))
central_aperture_F606W_217.plot(color='yellow', lw=2)
random_apertures_217_F606W.plot(color='lime', lw=2)
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlim((int(np.floor(x_min_217_F606W)), int(np.ceil(x_max_217_F606W))))
plt.ylim((int(np.floor(y_min_217_F606W)), int(np.ceil(y_max_217_F606W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F606W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F606W, interval=ZScaleInterval()))
central_aperture_F606W_481.plot(color='yellow', lw=2)
random_apertures_481_F606W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlim((int(np.floor(x_min_481_F606W)), int(np.ceil(x_max_481_F606W))))
plt.ylim((int(np.floor(y_min_481_F606W)), int(np.ceil(y_max_481_F606W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F606W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F606W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
for i, ax in enumerate(axs):
    axs[i].legend(
        ['Apertura del objeto', 'Apertura de puntos azarosos'], loc='upper right')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Lectura archivos .fits de filtro F814W
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

hdul_F814W = fits.open(
    '/home/javier/Escritorio/TTB/nircam1/egs_all_acs_wfc_f814w_030mas_v1.9_nircam1_mef.fits')
hdr_F814W = hdul_F814W[0].header
data_F814W = hdul_F814W[1].data

pixel_scale = 0.03  # arcsecs/pixel


"""""""""""""""""""""""""""""""""""""""""""""
Generación de apertura y anillos de fondo
"""""""""""""""""""""""""""""""""""""""""""""

# CEERSYJ-0012159472

aperture_diameter_arcsecs_472 = 0.3  # 0.35
aperture_radius_472 = aperture_diameter_arcsecs_472 / pixel_scale / 2

xc_472_F814W, yc_472_F814W = 2563.17, 2677.76
position_472_F814W = (xc_472_F814W, yc_472_F814W)

central_aperture_F814W_472 = CircularAperture(
    position_472_F814W, r=aperture_radius_472)
annulus_aperture_F814W_472 = CircularAnnulus(
    position_472_F814W, r_in=aperture_radius_472*2, r_out=aperture_radius_472*3)

delta_arcsecs_472_F814W = 3
delta_pix_472_F814W = delta_arcsecs_472_F814W / pixel_scale
x_min_472_F814W = xc_472_F814W - delta_pix_472_F814W
x_max_472_F814W = xc_472_F814W + delta_pix_472_F814W
y_min_472_F814W = yc_472_F814W - delta_pix_472_F814W
y_max_472_F814W = yc_472_F814W + delta_pix_472_F814W

# CEERSYJ-9586559217

aperture_diameter_arcsecs_217 = 0.3  # 0.208
aperture_radius_217 = aperture_diameter_arcsecs_217 / pixel_scale / 2

xc_217_F814W, yc_217_F814W = 3708.89, 2710.52
position_217_F814W = (xc_217_F814W, yc_217_F814W)

central_aperture_F814W_217 = CircularAperture(
    position_217_F814W, r=aperture_radius_217)
annulus_aperture_F814W_217 = CircularAnnulus(
    position_217_F814W, r_in=aperture_radius_217*2, r_out=aperture_radius_217*3)

delta_arcsecs_217_F814W = 3
delta_pix_217_F814W = delta_arcsecs_217_F814W / pixel_scale
x_min_217_F814W = xc_217_F814W - delta_pix_217_F814W
x_max_217_F814W = xc_217_F814W + delta_pix_217_F814W
y_min_217_F814W = yc_217_F814W - delta_pix_217_F814W
y_max_217_F814W = yc_217_F814W + delta_pix_217_F814W

# CEERSYJ-0012959481

aperture_diameter_arcsecs_481 = 0.3  # 0.336
aperture_radius_481 = aperture_diameter_arcsecs_481 / pixel_scale / 2

xc_481_F814W, yc_481_F814W = 2526.86, 2677.60
position_481_F814W = (xc_481_F814W, yc_481_F814W)

central_aperture_F814W_481 = CircularAperture(
    position_481_F814W, r=aperture_radius_481)
annulus_aperture_F814W_481 = CircularAnnulus(
    position_481_F814W, r_in=aperture_radius_481*2, r_out=aperture_radius_481*3)

delta_arcsecs_481_F814W = 3
delta_pix_481_F814W = delta_arcsecs_481_F814W / pixel_scale
x_min_481_F814W = xc_481_F814W - delta_pix_481_F814W
x_max_481_F814W = xc_481_F814W + delta_pix_481_F814W
y_min_481_F814W = yc_481_F814W - delta_pix_481_F814W
y_max_481_F814W = yc_481_F814W + delta_pix_481_F814W


"""""""""""""""""""""""""""""""""""""""
Gráfico de aperturas a las 3 galaxias
"""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F814W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F814W, interval=ZScaleInterval()))
central_aperture_F814W_472.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F814W_472.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_472_F814W, x_max_472_F814W, y_min_472_F814W, y_max_472_F814W])

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F814W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F814W, interval=ZScaleInterval()))
central_aperture_F814W_217.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F814W_217.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_217_F814W, x_max_217_F814W, y_min_217_F814W, y_max_217_F814W])

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F814W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F814W, interval=ZScaleInterval()))
central_aperture_F814W_481.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F814W_481.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_481_F814W, x_max_481_F814W, y_min_481_F814W, y_max_481_F814W])

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F814W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F814W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Fotometria de apertura con sustracción de fondo a CEERSYJ-0012159472,
CEERSYJ-9586559217 y CEERSYJ-0012959481 con una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

positions = [position_472_F814W, position_217_F814W, position_481_F814W]

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2

aperture = CircularAperture(positions, r=aperture_radius)
aperture_area = aperture.area

annulus_aperture = CircularAnnulus(
    positions, r_in=aperture_radius*2, r_out=aperture_radius*3)

aperstats = ApertureStats(data_F814W, annulus_aperture)
bkg_mean = aperstats.mean

total_bkg = bkg_mean * aperture_area

phot_table_F814W = aperture_photometry(data_F814W, aperture)
phot_bkgsub = phot_table_F814W['aperture_sum'] - total_bkg
phot_table_F814W['total_bkg'] = total_bkg
phot_table_F814W['aperture_sum_bkgsub'] = phot_bkgsub

for col in phot_table_F814W.colnames:
    phot_table_F814W[col].info.format = '%.8g'  # for consistent table output

print(phot_table_F814W)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Obtención de errores de la fotometría de apertura con
sustracción de fondo y para una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2
N_apertures = 50

# CEERSYJ-0012159472

cutout_472_F814W = data_F814W[int(np.floor(x_min_472_F814W)):int(np.ceil(
    x_max_472_F814W)), int(np.floor(y_min_472_F814W)):int(np.ceil(y_max_472_F814W))]

lower_x_472_F814W, upper_x_472_F814W = int(np.ceil(
    x_min_472_F814W + aperture_radius*3)), int(np.floor(x_max_472_F814W - aperture_radius*3))
lower_y_472_F814W, upper_y_472_F814W = int(np.ceil(
    y_min_472_F814W + aperture_radius*3)), int(np.floor(y_max_472_F814W - aperture_radius*3))

x_centers_472_F814W = [random.randrange(
    start=lower_x_472_F814W, stop=upper_x_472_F814W) for i in range(N_apertures)]
y_centers_472_F814W = [random.randrange(
    start=lower_y_472_F814W, stop=upper_y_472_F814W) for i in range(N_apertures)]

random_apertures_472_F814W = CircularAperture(
    zip(x_centers_472_F814W, y_centers_472_F814W), r=aperture_radius)
random_apertures_472_F814W_area = aperture.area

random_annulus_472_F814W = CircularAnnulus(zip(
    x_centers_472_F814W, y_centers_472_F814W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_472_F814W = ApertureStats(data_F814W, random_annulus_472_F814W)
annulus_bkg_mean_472_F814W = apertures_stats_472_F814W.mean

total_random_bkg_472_F814W = annulus_bkg_mean_472_F814W * \
    random_apertures_472_F814W_area

phot_table_472_F814W = aperture_photometry(
    data_F814W, random_apertures_472_F814W)
phot_bkgsub_472_F814W = phot_table_472_F814W['aperture_sum'] - \
    total_random_bkg_472_F814W
phot_table_472_F814W['total_bkg'] = total_random_bkg_472_F814W
phot_table_472_F814W['aperture_sum_bkgsub'] = phot_bkgsub_472_F814W

for col in phot_table_472_F814W.colnames:
    # for consistent table output
    phot_table_472_F814W[col].info.format = '%.8g'

print(phot_table_472_F814W)

fluxes_472_F814W = phot_table_472_F814W['aperture_sum_bkgsub']
print(len(fluxes_472_F814W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_472_F814W)))
fluxes_472_F814W = fluxes_472_F814W[np.where(
    fluxes_472_F814W < 1*np.std(fluxes_472_F814W))]
print(len(fluxes_472_F814W))
fluxes_472_F814W_std = np.std(fluxes_472_F814W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_472_F814W_std))

# CEERSYJ-9586559217

cutout_217_F814W = data_F814W[int(np.floor(x_min_217_F814W)):int(np.ceil(
    x_max_217_F814W)), int(np.floor(y_min_217_F814W)):int(np.ceil(y_max_217_F814W))]

lower_x_217_F814W, upper_x_217_F814W = int(np.ceil(
    x_min_217_F814W + aperture_radius*3)), int(np.floor(x_max_217_F814W - aperture_radius*3))
lower_y_217_F814W, upper_y_217_F814W = int(np.ceil(
    y_min_217_F814W + aperture_radius*3)), int(np.floor(y_max_217_F814W - aperture_radius*3))

x_centers_217_F814W = [random.randrange(
    start=lower_x_217_F814W, stop=upper_x_217_F814W) for i in range(N_apertures)]
y_centers_217_F814W = [random.randrange(
    start=lower_y_217_F814W, stop=upper_y_217_F814W) for i in range(N_apertures)]

random_apertures_217_F814W = CircularAperture(
    zip(x_centers_217_F814W, y_centers_217_F814W), r=aperture_radius)
random_apertures_217_F814W_area = aperture.area

random_annulus_217_F814W = CircularAnnulus(zip(
    x_centers_217_F814W, y_centers_217_F814W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_217_F814W = ApertureStats(data_F814W, random_annulus_217_F814W)
annulus_bkg_mean_217_F814W = apertures_stats_217_F814W.mean

total_random_bkg_217_F814W = annulus_bkg_mean_217_F814W * \
    random_apertures_217_F814W_area

phot_table_217_F814W = aperture_photometry(
    data_F814W, random_apertures_217_F814W)
phot_bkgsub_217_F814W = phot_table_217_F814W['aperture_sum'] - \
    total_random_bkg_217_F814W
phot_table_217_F814W['total_bkg'] = total_random_bkg_217_F814W
phot_table_217_F814W['aperture_sum_bkgsub'] = phot_bkgsub_217_F814W

for col in phot_table_217_F814W.colnames:
    # for consistent table output
    phot_table_217_F814W[col].info.format = '%.8g'

print(phot_table_217_F814W)

fluxes_217_F814W = phot_table_217_F814W['aperture_sum_bkgsub']
print(len(fluxes_217_F814W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_217_F814W)))
fluxes_217_F814W = fluxes_217_F814W[np.where(
    fluxes_217_F814W < 1*np.std(fluxes_217_F814W))]
print(len(fluxes_217_F814W))
fluxes_217_F814W_std = np.std(fluxes_217_F814W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_217_F814W_std))

# CEERSYJ-0012959481

cutout_481_F814W = data_F814W[int(np.floor(x_min_481_F814W)):int(np.ceil(
    x_max_481_F814W)), int(np.floor(y_min_481_F814W)):int(np.ceil(y_max_481_F814W))]

lower_x_481_F814W, upper_x_481_F814W = int(np.ceil(
    x_min_481_F814W + aperture_radius*3)), int(np.floor(x_max_481_F814W - aperture_radius*3))
lower_y_481_F814W, upper_y_481_F814W = int(np.ceil(
    y_min_481_F814W + aperture_radius*3)), int(np.floor(y_max_481_F814W - aperture_radius*3))

x_centers_481_F814W = [random.randrange(
    start=lower_x_481_F814W, stop=upper_x_481_F814W) for i in range(N_apertures)]
y_centers_481_F814W = [random.randrange(
    start=lower_y_481_F814W, stop=upper_y_481_F814W) for i in range(N_apertures)]

random_apertures_481_F814W = CircularAperture(
    zip(x_centers_481_F814W, y_centers_481_F814W), r=aperture_radius)
random_apertures_481_F814W_area = aperture.area

random_annulus_481_F814W = CircularAnnulus(zip(
    x_centers_481_F814W, y_centers_481_F814W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_481_F814W = ApertureStats(data_F814W, random_annulus_481_F814W)
annulus_bkg_mean_481_F814W = apertures_stats_481_F814W.mean

total_random_bkg_481_F814W = annulus_bkg_mean_481_F814W * \
    random_apertures_481_F814W_area

phot_table_481_F814W = aperture_photometry(
    data_F814W, random_apertures_481_F814W)
phot_bkgsub_481_F814W = phot_table_481_F814W['aperture_sum'] - \
    total_random_bkg_481_F814W
phot_table_481_F814W['total_bkg'] = total_random_bkg_481_F814W
phot_table_481_F814W['aperture_sum_bkgsub'] = phot_bkgsub_481_F814W

for col in phot_table_481_F814W.colnames:
    # for consistent table output
    phot_table_481_F814W[col].info.format = '%.8g'

print(phot_table_481_F814W)

fluxes_481_F814W = phot_table_481_F814W['aperture_sum_bkgsub']
print(len(fluxes_481_F814W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_481_F814W)))
fluxes_481_F814W = fluxes_481_F814W[np.where(
    fluxes_481_F814W < 1*np.std(fluxes_481_F814W))]
print(len(fluxes_481_F814W))
fluxes_481_F814W_std = np.std(fluxes_481_F814W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_481_F814W_std))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Gráfico de apertura de puntos azarosos cercanos a
CEERSYJ-0012159472, CEERSYJ-9586559217 y CEERSYJ-0012959481
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F814W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F814W, interval=ZScaleInterval()))
central_aperture_F814W_472.plot(color='yellow', lw=2)
random_apertures_472_F814W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlim((int(np.floor(x_min_472_F814W)), int(np.ceil(x_max_472_F814W))))
plt.ylim((int(np.floor(y_min_472_F814W)), int(np.ceil(y_max_472_F814W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F814W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F814W, interval=ZScaleInterval()))
central_aperture_F814W_217.plot(color='yellow', lw=2)
random_apertures_217_F814W.plot(color='lime', lw=2)
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlim((int(np.floor(x_min_217_F814W)), int(np.ceil(x_max_217_F814W))))
plt.ylim((int(np.floor(y_min_217_F814W)), int(np.ceil(y_max_217_F814W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F814W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F814W, interval=ZScaleInterval()))
central_aperture_F814W_481.plot(color='yellow', lw=2)
random_apertures_481_F814W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlim((int(np.floor(x_min_481_F814W)), int(np.ceil(x_max_481_F814W))))
plt.ylim((int(np.floor(y_min_481_F814W)), int(np.ceil(y_max_481_F814W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F814W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F814W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
for i, ax in enumerate(axs):
    axs[i].legend(
        ['Apertura del objeto', 'Apertura de puntos azarosos'], loc='upper right')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Lectura archivos .fits de filtro F115W
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

hdul_F115W = fits.open(
    '/home/javier/Escritorio/TTB/nircam1/hlsp_ceers_jwst_nircam_nircam1_f115w_dr0.5_i2d.fits')
hdr_F115W = hdul_F115W[0].header
data_F115W = hdul_F115W[1].data

pixel_scale = 0.03  # arcsecs/pixel


"""""""""""""""""""""""""""""""""""""""""""""
Generación de apertura y anillos de fondo
"""""""""""""""""""""""""""""""""""""""""""""

# CEERSYJ-0012159472

aperture_diameter_arcsecs_472 = 0.3  # 0.35
aperture_radius_472 = aperture_diameter_arcsecs_472 / pixel_scale / 2

xc_472_F115W, yc_472_F115W = 2563.17, 2677.76
position_472_F115W = (xc_472_F115W, yc_472_F115W)

central_aperture_F115W_472 = CircularAperture(
    position_472_F115W, r=aperture_radius_472)
annulus_aperture_F115W_472 = CircularAnnulus(
    position_472_F115W, r_in=aperture_radius_472*2, r_out=aperture_radius_472*3)

delta_arcsecs_472_F115W = 3
delta_pix_472_F115W = delta_arcsecs_472_F115W / pixel_scale
x_min_472_F115W = xc_472_F115W - delta_pix_472_F115W
x_max_472_F115W = xc_472_F115W + delta_pix_472_F115W
y_min_472_F115W = yc_472_F115W - delta_pix_472_F115W
y_max_472_F115W = yc_472_F115W + delta_pix_472_F115W

# CEERSYJ-9586559217

aperture_diameter_arcsecs_217 = 0.3  # 0.208
aperture_radius_217 = aperture_diameter_arcsecs_217 / pixel_scale / 2

xc_217_F115W, yc_217_F115W = 3708.89, 2710.52
position_217_F115W = (xc_217_F115W, yc_217_F115W)

central_aperture_F115W_217 = CircularAperture(
    position_217_F115W, r=aperture_radius_217)
annulus_aperture_F115W_217 = CircularAnnulus(
    position_217_F115W, r_in=aperture_radius_217*2, r_out=aperture_radius_217*3)

delta_arcsecs_217_F115W = 3
delta_pix_217_F115W = delta_arcsecs_217_F115W / pixel_scale
x_min_217_F115W = xc_217_F115W - delta_pix_217_F115W
x_max_217_F115W = xc_217_F115W + delta_pix_217_F115W
y_min_217_F115W = yc_217_F115W - delta_pix_217_F115W
y_max_217_F115W = yc_217_F115W + delta_pix_217_F115W

# CEERSYJ-0012959481

aperture_diameter_arcsecs_481 = 0.3  # 0.336
aperture_radius_481 = aperture_diameter_arcsecs_481 / pixel_scale / 2

xc_481_F115W, yc_481_F115W = 2526.86, 2677.60
position_481_F115W = (xc_481_F115W, yc_481_F115W)

central_aperture_F115W_481 = CircularAperture(
    position_481_F115W, r=aperture_radius_481)
annulus_aperture_F115W_481 = CircularAnnulus(
    position_481_F115W, r_in=aperture_radius_481*2, r_out=aperture_radius_481*3)

delta_arcsecs_481_F115W = 3
delta_pix_481_F115W = delta_arcsecs_481_F115W / pixel_scale
x_min_481_F115W = xc_481_F115W - delta_pix_481_F115W
x_max_481_F115W = xc_481_F115W + delta_pix_481_F115W
y_min_481_F115W = yc_481_F115W - delta_pix_481_F115W
y_max_481_F115W = yc_481_F115W + delta_pix_481_F115W


"""""""""""""""""""""""""""""""""""""""
Gráfico de aperturas a las 3 galaxias
"""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F115W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F115W, interval=ZScaleInterval()))
central_aperture_F115W_472.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F115W_472.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_472_F115W, x_max_472_F115W, y_min_472_F115W, y_max_472_F115W])

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F115W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F115W, interval=ZScaleInterval()))
central_aperture_F115W_217.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F115W_217.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_217_F115W, x_max_217_F115W, y_min_217_F115W, y_max_217_F115W])

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F115W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F115W, interval=ZScaleInterval()))
central_aperture_F115W_481.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F115W_481.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_481_F115W, x_max_481_F115W, y_min_481_F115W, y_max_481_F115W])

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F115W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F115W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Fotometria de apertura con sustracción de fondo a CEERSYJ-0012159472,
CEERSYJ-9586559217 y CEERSYJ-0012959481 con una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

positions = [position_472_F115W, position_217_F115W, position_481_F115W]

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2

aperture = CircularAperture(positions, r=aperture_radius)
aperture_area = aperture.area

annulus_aperture = CircularAnnulus(
    positions, r_in=aperture_radius*2, r_out=aperture_radius*3)

aperstats = ApertureStats(data_F115W, annulus_aperture)
bkg_mean = aperstats.mean

total_bkg = bkg_mean * aperture_area

phot_table_F115W = aperture_photometry(data_F115W, aperture)
phot_bkgsub = phot_table_F115W['aperture_sum'] - total_bkg
phot_table_F115W['total_bkg'] = total_bkg
phot_table_F115W['aperture_sum_bkgsub'] = phot_bkgsub

for col in phot_table_F115W.colnames:
    phot_table_F115W[col].info.format = '%.8g'  # for consistent table output

print(phot_table_F115W)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Obtención de errores de la fotometría de apertura con
sustracción de fondo y para una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2
N_apertures = 50

# CEERSYJ-0012159472

cutout_472_F115W = data_F115W[int(np.floor(x_min_472_F115W)):int(np.ceil(
    x_max_472_F115W)), int(np.floor(y_min_472_F115W)):int(np.ceil(y_max_472_F115W))]

lower_x_472_F115W, upper_x_472_F115W = int(np.ceil(
    x_min_472_F115W + aperture_radius*3)), int(np.floor(x_max_472_F115W - aperture_radius*3))
lower_y_472_F115W, upper_y_472_F115W = int(np.ceil(
    y_min_472_F115W + aperture_radius*3)), int(np.floor(y_max_472_F115W - aperture_radius*3))

x_centers_472_F115W = [random.randrange(
    start=lower_x_472_F115W, stop=upper_x_472_F115W) for i in range(N_apertures)]
y_centers_472_F115W = [random.randrange(
    start=lower_y_472_F115W, stop=upper_y_472_F115W) for i in range(N_apertures)]

random_apertures_472_F115W = CircularAperture(
    zip(x_centers_472_F115W, y_centers_472_F115W), r=aperture_radius)
random_apertures_472_F115W_area = aperture.area

random_annulus_472_F115W = CircularAnnulus(zip(
    x_centers_472_F115W, y_centers_472_F115W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_472_F115W = ApertureStats(data_F115W, random_annulus_472_F115W)
annulus_bkg_mean_472_F115W = apertures_stats_472_F115W.mean

total_random_bkg_472_F115W = annulus_bkg_mean_472_F115W * \
    random_apertures_472_F115W_area

phot_table_472_F115W = aperture_photometry(
    data_F115W, random_apertures_472_F115W)
phot_bkgsub_472_F115W = phot_table_472_F115W['aperture_sum'] - \
    total_random_bkg_472_F115W
phot_table_472_F115W['total_bkg'] = total_random_bkg_472_F115W
phot_table_472_F115W['aperture_sum_bkgsub'] = phot_bkgsub_472_F115W

for col in phot_table_472_F115W.colnames:
    # for consistent table output
    phot_table_472_F115W[col].info.format = '%.8g'

print(phot_table_472_F115W)

fluxes_472_F115W = phot_table_472_F115W['aperture_sum_bkgsub']
print(len(fluxes_472_F115W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_472_F115W)))
fluxes_472_F115W = fluxes_472_F115W[np.where(
    fluxes_472_F115W < 1*np.std(fluxes_472_F115W))]
print(len(fluxes_472_F115W))
fluxes_472_F115W_std = np.std(fluxes_472_F115W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_472_F115W_std))

# CEERSYJ-9586559217

cutout_217_F115W = data_F115W[int(np.floor(x_min_217_F115W)):int(np.ceil(
    x_max_217_F115W)), int(np.floor(y_min_217_F115W)):int(np.ceil(y_max_217_F115W))]

lower_x_217_F115W, upper_x_217_F115W = int(np.ceil(
    x_min_217_F115W + aperture_radius*3)), int(np.floor(x_max_217_F115W - aperture_radius*3))
lower_y_217_F115W, upper_y_217_F115W = int(np.ceil(
    y_min_217_F115W + aperture_radius*3)), int(np.floor(y_max_217_F115W - aperture_radius*3))

x_centers_217_F115W = [random.randrange(
    start=lower_x_217_F115W, stop=upper_x_217_F115W) for i in range(N_apertures)]
y_centers_217_F115W = [random.randrange(
    start=lower_y_217_F115W, stop=upper_y_217_F115W) for i in range(N_apertures)]

random_apertures_217_F115W = CircularAperture(
    zip(x_centers_217_F115W, y_centers_217_F115W), r=aperture_radius)
random_apertures_217_F115W_area = aperture.area

random_annulus_217_F115W = CircularAnnulus(zip(
    x_centers_217_F115W, y_centers_217_F115W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_217_F115W = ApertureStats(data_F115W, random_annulus_217_F115W)
annulus_bkg_mean_217_F115W = apertures_stats_217_F115W.mean

total_random_bkg_217_F115W = annulus_bkg_mean_217_F115W * \
    random_apertures_217_F115W_area

phot_table_217_F115W = aperture_photometry(
    data_F115W, random_apertures_217_F115W)
phot_bkgsub_217_F115W = phot_table_217_F115W['aperture_sum'] - \
    total_random_bkg_217_F115W
phot_table_217_F115W['total_bkg'] = total_random_bkg_217_F115W
phot_table_217_F115W['aperture_sum_bkgsub'] = phot_bkgsub_217_F115W

for col in phot_table_217_F115W.colnames:
    # for consistent table output
    phot_table_217_F115W[col].info.format = '%.8g'

print(phot_table_217_F115W)

fluxes_217_F115W = phot_table_217_F115W['aperture_sum_bkgsub']
print(len(fluxes_217_F115W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_217_F115W)))
fluxes_217_F115W = fluxes_217_F115W[np.where(
    fluxes_217_F115W < 1*np.std(fluxes_217_F115W))]
print(len(fluxes_217_F115W))
fluxes_217_F115W_std = np.std(fluxes_217_F115W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_217_F115W_std))

# CEERSYJ-0012959481

cutout_481_F115W = data_F115W[int(np.floor(x_min_481_F115W)):int(np.ceil(
    x_max_481_F115W)), int(np.floor(y_min_481_F115W)):int(np.ceil(y_max_481_F115W))]

lower_x_481_F115W, upper_x_481_F115W = int(np.ceil(
    x_min_481_F115W + aperture_radius*3)), int(np.floor(x_max_481_F115W - aperture_radius*3))
lower_y_481_F115W, upper_y_481_F115W = int(np.ceil(
    y_min_481_F115W + aperture_radius*3)), int(np.floor(y_max_481_F115W - aperture_radius*3))

x_centers_481_F115W = [random.randrange(
    start=lower_x_481_F115W, stop=upper_x_481_F115W) for i in range(N_apertures)]
y_centers_481_F115W = [random.randrange(
    start=lower_y_481_F115W, stop=upper_y_481_F115W) for i in range(N_apertures)]

random_apertures_481_F115W = CircularAperture(
    zip(x_centers_481_F115W, y_centers_481_F115W), r=aperture_radius)
random_apertures_481_F115W_area = aperture.area

random_annulus_481_F115W = CircularAnnulus(zip(
    x_centers_481_F115W, y_centers_481_F115W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_481_F115W = ApertureStats(data_F115W, random_annulus_481_F115W)
annulus_bkg_mean_481_F115W = apertures_stats_481_F115W.mean

total_random_bkg_481_F115W = annulus_bkg_mean_481_F115W * \
    random_apertures_481_F115W_area

phot_table_481_F115W = aperture_photometry(
    data_F115W, random_apertures_481_F115W)
phot_bkgsub_481_F115W = phot_table_481_F115W['aperture_sum'] - \
    total_random_bkg_481_F115W
phot_table_481_F115W['total_bkg'] = total_random_bkg_481_F115W
phot_table_481_F115W['aperture_sum_bkgsub'] = phot_bkgsub_481_F115W

for col in phot_table_481_F115W.colnames:
    # for consistent table output
    phot_table_481_F115W[col].info.format = '%.8g'

print(phot_table_481_F115W)

fluxes_481_F115W = phot_table_481_F115W['aperture_sum_bkgsub']
print(len(fluxes_481_F115W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_481_F115W)))
fluxes_481_F115W = fluxes_481_F115W[np.where(
    fluxes_481_F115W < 1*np.std(fluxes_481_F115W))]
print(len(fluxes_481_F115W))
fluxes_481_F115W_std = np.std(fluxes_481_F115W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_481_F115W_std))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Gráfico de apertura de puntos azarosos cercanos a
CEERSYJ-0012159472, CEERSYJ-9586559217 y CEERSYJ-0012959481
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F115W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F115W, interval=ZScaleInterval()))
central_aperture_F115W_472.plot(color='yellow', lw=2)
random_apertures_472_F115W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlim((int(np.floor(x_min_472_F115W)), int(np.ceil(x_max_472_F115W))))
plt.ylim((int(np.floor(y_min_472_F115W)), int(np.ceil(y_max_472_F115W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F115W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F115W, interval=ZScaleInterval()))
central_aperture_F115W_217.plot(color='yellow', lw=2)
random_apertures_217_F115W.plot(color='lime', lw=2)
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlim((int(np.floor(x_min_217_F115W)), int(np.ceil(x_max_217_F115W))))
plt.ylim((int(np.floor(y_min_217_F115W)), int(np.ceil(y_max_217_F115W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F115W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F115W, interval=ZScaleInterval()))
central_aperture_F115W_481.plot(color='yellow', lw=2)
random_apertures_481_F115W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlim((int(np.floor(x_min_481_F115W)), int(np.ceil(x_max_481_F115W))))
plt.ylim((int(np.floor(y_min_481_F115W)), int(np.ceil(y_max_481_F115W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F115W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F115W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
for i, ax in enumerate(axs):
    axs[i].legend(
        ['Apertura del objeto', 'Apertura de puntos azarosos'], loc='upper right')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Lectura archivos .fits de filtro F125W
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

hdul_F125W = fits.open(
    '/home/javier/Escritorio/TTB/nircam1/egs_all_wfc3_ir_f125w_030mas_v1.9_nircam1_mef.fits')
hdr_F125W = hdul_F125W[0].header
data_F125W = hdul_F125W[1].data

pixel_scale = 0.03  # arcsecs/pixel


"""""""""""""""""""""""""""""""""""""""""""""
Generación de apertura y anillos de fondo
"""""""""""""""""""""""""""""""""""""""""""""

# CEERSYJ-0012159472

aperture_diameter_arcsecs_472 = 0.3  # 0.35
aperture_radius_472 = aperture_diameter_arcsecs_472 / pixel_scale / 2

xc_472_F125W, yc_472_F125W = 2563.17, 2677.76
position_472_F125W = (xc_472_F125W, yc_472_F125W)

central_aperture_F125W_472 = CircularAperture(
    position_472_F125W, r=aperture_radius_472)
annulus_aperture_F125W_472 = CircularAnnulus(
    position_472_F125W, r_in=aperture_radius_472*2, r_out=aperture_radius_472*3)

delta_arcsecs_472_F125W = 3
delta_pix_472_F125W = delta_arcsecs_472_F125W / pixel_scale
x_min_472_F125W = xc_472_F125W - delta_pix_472_F125W
x_max_472_F125W = xc_472_F125W + delta_pix_472_F125W
y_min_472_F125W = yc_472_F125W - delta_pix_472_F125W
y_max_472_F125W = yc_472_F125W + delta_pix_472_F125W

# CEERSYJ-9586559217

aperture_diameter_arcsecs_217 = 0.3  # 0.208
aperture_radius_217 = aperture_diameter_arcsecs_217 / pixel_scale / 2

xc_217_F125W, yc_217_F125W = 3708.89, 2710.52
position_217_F125W = (xc_217_F125W, yc_217_F125W)

central_aperture_F125W_217 = CircularAperture(
    position_217_F125W, r=aperture_radius_217)
annulus_aperture_F125W_217 = CircularAnnulus(
    position_217_F125W, r_in=aperture_radius_217*2, r_out=aperture_radius_217*3)

delta_arcsecs_217_F125W = 3
delta_pix_217_F125W = delta_arcsecs_217_F125W / pixel_scale
x_min_217_F125W = xc_217_F125W - delta_pix_217_F125W
x_max_217_F125W = xc_217_F125W + delta_pix_217_F125W
y_min_217_F125W = yc_217_F125W - delta_pix_217_F125W
y_max_217_F125W = yc_217_F125W + delta_pix_217_F125W

# CEERSYJ-0012959481

aperture_diameter_arcsecs_481 = 0.3  # 0.336
aperture_radius_481 = aperture_diameter_arcsecs_481 / pixel_scale / 2

xc_481_F125W, yc_481_F125W = 2526.86, 2677.60
position_481_F125W = (xc_481_F125W, yc_481_F125W)

central_aperture_F125W_481 = CircularAperture(
    position_481_F125W, r=aperture_radius_481)
annulus_aperture_F125W_481 = CircularAnnulus(
    position_481_F125W, r_in=aperture_radius_481*2, r_out=aperture_radius_481*3)

delta_arcsecs_481_F125W = 3
delta_pix_481_F125W = delta_arcsecs_481_F125W / pixel_scale
x_min_481_F125W = xc_481_F125W - delta_pix_481_F125W
x_max_481_F125W = xc_481_F125W + delta_pix_481_F125W
y_min_481_F125W = yc_481_F125W - delta_pix_481_F125W
y_max_481_F125W = yc_481_F125W + delta_pix_481_F125W


"""""""""""""""""""""""""""""""""""""""
Gráfico de aperturas a las 3 galaxias
"""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F125W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F125W, interval=ZScaleInterval()))
central_aperture_F125W_472.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F125W_472.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_472_F125W, x_max_472_F125W, y_min_472_F125W, y_max_472_F125W])

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F125W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F125W, interval=ZScaleInterval()))
central_aperture_F125W_217.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F125W_217.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_217_F125W, x_max_217_F125W, y_min_217_F125W, y_max_217_F125W])

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F125W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F125W, interval=ZScaleInterval()))
central_aperture_F125W_481.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F125W_481.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_481_F125W, x_max_481_F125W, y_min_481_F125W, y_max_481_F125W])

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F125W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F125W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Fotometria de apertura con sustracción de fondo a CEERSYJ-0012159472,
CEERSYJ-9586559217 y CEERSYJ-0012959481 con una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

positions = [position_472_F125W, position_217_F125W, position_481_F125W]

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2

aperture = CircularAperture(positions, r=aperture_radius)
aperture_area = aperture.area

annulus_aperture = CircularAnnulus(
    positions, r_in=aperture_radius*2, r_out=aperture_radius*3)

aperstats = ApertureStats(data_F125W, annulus_aperture)
bkg_mean = aperstats.mean

total_bkg = bkg_mean * aperture_area

phot_table_F125W = aperture_photometry(data_F125W, aperture)
phot_bkgsub = phot_table_F125W['aperture_sum'] - total_bkg
phot_table_F125W['total_bkg'] = total_bkg
phot_table_F125W['aperture_sum_bkgsub'] = phot_bkgsub

for col in phot_table_F125W.colnames:
    phot_table_F125W[col].info.format = '%.8g'  # for consistent table output

print(phot_table_F125W)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Obtención de errores de la fotometría de apertura con
sustracción de fondo y para una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2
N_apertures = 50

# CEERSYJ-0012159472

cutout_472_F125W = data_F125W[int(np.floor(x_min_472_F125W)):int(np.ceil(
    x_max_472_F125W)), int(np.floor(y_min_472_F125W)):int(np.ceil(y_max_472_F125W))]

lower_x_472_F125W, upper_x_472_F125W = int(np.ceil(
    x_min_472_F125W + aperture_radius*3)), int(np.floor(x_max_472_F125W - aperture_radius*3))
lower_y_472_F125W, upper_y_472_F125W = int(np.ceil(
    y_min_472_F125W + aperture_radius*3)), int(np.floor(y_max_472_F125W - aperture_radius*3))

x_centers_472_F125W = [random.randrange(
    start=lower_x_472_F125W, stop=upper_x_472_F125W) for i in range(N_apertures)]
y_centers_472_F125W = [random.randrange(
    start=lower_y_472_F125W, stop=upper_y_472_F125W) for i in range(N_apertures)]

random_apertures_472_F125W = CircularAperture(
    zip(x_centers_472_F125W, y_centers_472_F125W), r=aperture_radius)
random_apertures_472_F125W_area = aperture.area

random_annulus_472_F125W = CircularAnnulus(zip(
    x_centers_472_F125W, y_centers_472_F125W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_472_F125W = ApertureStats(data_F125W, random_annulus_472_F125W)
annulus_bkg_mean_472_F125W = apertures_stats_472_F125W.mean

total_random_bkg_472_F125W = annulus_bkg_mean_472_F125W * \
    random_apertures_472_F125W_area

phot_table_472_F125W = aperture_photometry(
    data_F125W, random_apertures_472_F125W)
phot_bkgsub_472_F125W = phot_table_472_F125W['aperture_sum'] - \
    total_random_bkg_472_F125W
phot_table_472_F125W['total_bkg'] = total_random_bkg_472_F125W
phot_table_472_F125W['aperture_sum_bkgsub'] = phot_bkgsub_472_F125W

for col in phot_table_472_F125W.colnames:
    # for consistent table output
    phot_table_472_F125W[col].info.format = '%.8g'

print(phot_table_472_F125W)

fluxes_472_F125W = phot_table_472_F125W['aperture_sum_bkgsub']
print(len(fluxes_472_F125W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_472_F125W)))
fluxes_472_F125W = fluxes_472_F125W[np.where(
    fluxes_472_F125W < 1*np.std(fluxes_472_F125W))]
print(len(fluxes_472_F125W))
fluxes_472_F125W_std = np.std(fluxes_472_F125W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_472_F125W_std))

# CEERSYJ-9586559217

cutout_217_F125W = data_F125W[int(np.floor(x_min_217_F125W)):int(np.ceil(
    x_max_217_F125W)), int(np.floor(y_min_217_F125W)):int(np.ceil(y_max_217_F125W))]

lower_x_217_F125W, upper_x_217_F125W = int(np.ceil(
    x_min_217_F125W + aperture_radius*3)), int(np.floor(x_max_217_F125W - aperture_radius*3))
lower_y_217_F125W, upper_y_217_F125W = int(np.ceil(
    y_min_217_F125W + aperture_radius*3)), int(np.floor(y_max_217_F125W - aperture_radius*3))

x_centers_217_F125W = [random.randrange(
    start=lower_x_217_F125W, stop=upper_x_217_F125W) for i in range(N_apertures)]
y_centers_217_F125W = [random.randrange(
    start=lower_y_217_F125W, stop=upper_y_217_F125W) for i in range(N_apertures)]

random_apertures_217_F125W = CircularAperture(
    zip(x_centers_217_F125W, y_centers_217_F125W), r=aperture_radius)
random_apertures_217_F125W_area = aperture.area

random_annulus_217_F125W = CircularAnnulus(zip(
    x_centers_217_F125W, y_centers_217_F125W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_217_F125W = ApertureStats(data_F125W, random_annulus_217_F125W)
annulus_bkg_mean_217_F125W = apertures_stats_217_F125W.mean

total_random_bkg_217_F125W = annulus_bkg_mean_217_F125W * \
    random_apertures_217_F125W_area

phot_table_217_F125W = aperture_photometry(
    data_F125W, random_apertures_217_F125W)
phot_bkgsub_217_F125W = phot_table_217_F125W['aperture_sum'] - \
    total_random_bkg_217_F125W
phot_table_217_F125W['total_bkg'] = total_random_bkg_217_F125W
phot_table_217_F125W['aperture_sum_bkgsub'] = phot_bkgsub_217_F125W

for col in phot_table_217_F125W.colnames:
    # for consistent table output
    phot_table_217_F125W[col].info.format = '%.8g'

print(phot_table_217_F125W)

fluxes_217_F125W = phot_table_217_F125W['aperture_sum_bkgsub']
print(len(fluxes_217_F125W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_217_F125W)))
fluxes_217_F125W = fluxes_217_F125W[np.where(
    fluxes_217_F125W < 1*np.std(fluxes_217_F125W))]
print(len(fluxes_217_F125W))
fluxes_217_F125W_std = np.std(fluxes_217_F125W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_217_F125W_std))

# CEERSYJ-0012959481

cutout_481_F125W = data_F125W[int(np.floor(x_min_481_F125W)):int(np.ceil(
    x_max_481_F125W)), int(np.floor(y_min_481_F125W)):int(np.ceil(y_max_481_F125W))]

lower_x_481_F125W, upper_x_481_F125W = int(np.ceil(
    x_min_481_F125W + aperture_radius*3)), int(np.floor(x_max_481_F125W - aperture_radius*3))
lower_y_481_F125W, upper_y_481_F125W = int(np.ceil(
    y_min_481_F125W + aperture_radius*3)), int(np.floor(y_max_481_F125W - aperture_radius*3))

x_centers_481_F125W = [random.randrange(
    start=lower_x_481_F125W, stop=upper_x_481_F125W) for i in range(N_apertures)]
y_centers_481_F125W = [random.randrange(
    start=lower_y_481_F125W, stop=upper_y_481_F125W) for i in range(N_apertures)]

random_apertures_481_F125W = CircularAperture(
    zip(x_centers_481_F125W, y_centers_481_F125W), r=aperture_radius)
random_apertures_481_F125W_area = aperture.area

random_annulus_481_F125W = CircularAnnulus(zip(
    x_centers_481_F125W, y_centers_481_F125W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_481_F125W = ApertureStats(data_F125W, random_annulus_481_F125W)
annulus_bkg_mean_481_F125W = apertures_stats_481_F125W.mean

total_random_bkg_481_F125W = annulus_bkg_mean_481_F125W * \
    random_apertures_481_F125W_area

phot_table_481_F125W = aperture_photometry(
    data_F125W, random_apertures_481_F125W)
phot_bkgsub_481_F125W = phot_table_481_F125W['aperture_sum'] - \
    total_random_bkg_481_F125W
phot_table_481_F125W['total_bkg'] = total_random_bkg_481_F125W
phot_table_481_F125W['aperture_sum_bkgsub'] = phot_bkgsub_481_F125W

for col in phot_table_481_F125W.colnames:
    # for consistent table output
    phot_table_481_F125W[col].info.format = '%.8g'

print(phot_table_481_F125W)

fluxes_481_F125W = phot_table_481_F125W['aperture_sum_bkgsub']
print(len(fluxes_481_F125W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_481_F125W)))
fluxes_481_F125W = fluxes_481_F125W[np.where(
    fluxes_481_F125W < 1*np.std(fluxes_481_F125W))]
print(len(fluxes_481_F125W))
fluxes_481_F125W_std = np.std(fluxes_481_F125W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_481_F125W_std))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Gráfico de apertura de puntos azarosos cercanos a
CEERSYJ-0012159472, CEERSYJ-9586559217 y CEERSYJ-0012959481
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F125W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F125W, interval=ZScaleInterval()))
central_aperture_F125W_472.plot(color='yellow', lw=2)
random_apertures_472_F125W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlim((int(np.floor(x_min_472_F125W)), int(np.ceil(x_max_472_F125W))))
plt.ylim((int(np.floor(y_min_472_F125W)), int(np.ceil(y_max_472_F125W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F125W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F125W, interval=ZScaleInterval()))
central_aperture_F125W_217.plot(color='yellow', lw=2)
random_apertures_217_F125W.plot(color='lime', lw=2)
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlim((int(np.floor(x_min_217_F125W)), int(np.ceil(x_max_217_F125W))))
plt.ylim((int(np.floor(y_min_217_F125W)), int(np.ceil(y_max_217_F125W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F125W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F125W, interval=ZScaleInterval()))
central_aperture_F125W_481.plot(color='yellow', lw=2)
random_apertures_481_F125W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlim((int(np.floor(x_min_481_F125W)), int(np.ceil(x_max_481_F125W))))
plt.ylim((int(np.floor(y_min_481_F125W)), int(np.ceil(y_max_481_F125W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F125W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F125W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
for i, ax in enumerate(axs):
    axs[i].legend(
        ['Apertura del objeto', 'Apertura de puntos azarosos'], loc='upper right')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Lectura archivos .fits de filtro F140W
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

hdul_F140W = fits.open(
    '/home/javier/Escritorio/TTB/nircam1/egs_all_wfc3_ir_f140w_030mas_v1.9_nircam1_mef.fits')
hdr_F140W = hdul_F140W[0].header
data_F140W = hdul_F140W[1].data

pixel_scale = 0.03  # arcsecs/pixel


"""""""""""""""""""""""""""""""""""""""""""""
Generación de apertura y anillos de fondo
"""""""""""""""""""""""""""""""""""""""""""""

# CEERSYJ-0012159472

aperture_diameter_arcsecs_472 = 0.3  # 0.35
aperture_radius_472 = aperture_diameter_arcsecs_472 / pixel_scale / 2

xc_472_F140W, yc_472_F140W = 2563.17, 2677.76
position_472_F140W = (xc_472_F140W, yc_472_F140W)

central_aperture_F140W_472 = CircularAperture(
    position_472_F140W, r=aperture_radius_472)
annulus_aperture_F140W_472 = CircularAnnulus(
    position_472_F140W, r_in=aperture_radius_472*2, r_out=aperture_radius_472*3)

delta_arcsecs_472_F140W = 3
delta_pix_472_F140W = delta_arcsecs_472_F140W / pixel_scale
x_min_472_F140W = xc_472_F140W - delta_pix_472_F140W
x_max_472_F140W = xc_472_F140W + delta_pix_472_F140W
y_min_472_F140W = yc_472_F140W - delta_pix_472_F140W
y_max_472_F140W = yc_472_F140W + delta_pix_472_F140W

# CEERSYJ-9586559217

aperture_diameter_arcsecs_217 = 0.3  # 0.208
aperture_radius_217 = aperture_diameter_arcsecs_217 / pixel_scale / 2

xc_217_F140W, yc_217_F140W = 3708.89, 2710.52
position_217_F140W = (xc_217_F140W, yc_217_F140W)

central_aperture_F140W_217 = CircularAperture(
    position_217_F140W, r=aperture_radius_217)
annulus_aperture_F140W_217 = CircularAnnulus(
    position_217_F140W, r_in=aperture_radius_217*2, r_out=aperture_radius_217*3)

delta_arcsecs_217_F140W = 3
delta_pix_217_F140W = delta_arcsecs_217_F140W / pixel_scale
x_min_217_F140W = xc_217_F140W - delta_pix_217_F140W
x_max_217_F140W = xc_217_F140W + delta_pix_217_F140W
y_min_217_F140W = yc_217_F140W - delta_pix_217_F140W
y_max_217_F140W = yc_217_F140W + delta_pix_217_F140W

# CEERSYJ-0012959481

aperture_diameter_arcsecs_481 = 0.3  # 0.336
aperture_radius_481 = aperture_diameter_arcsecs_481 / pixel_scale / 2

xc_481_F140W, yc_481_F140W = 2526.86, 2677.60
position_481_F140W = (xc_481_F140W, yc_481_F140W)

central_aperture_F140W_481 = CircularAperture(
    position_481_F140W, r=aperture_radius_481)
annulus_aperture_F140W_481 = CircularAnnulus(
    position_481_F140W, r_in=aperture_radius_481*2, r_out=aperture_radius_481*3)

delta_arcsecs_481_F140W = 3
delta_pix_481_F140W = delta_arcsecs_481_F140W / pixel_scale
x_min_481_F140W = xc_481_F140W - delta_pix_481_F140W
x_max_481_F140W = xc_481_F140W + delta_pix_481_F140W
y_min_481_F140W = yc_481_F140W - delta_pix_481_F140W
y_max_481_F140W = yc_481_F140W + delta_pix_481_F140W


"""""""""""""""""""""""""""""""""""""""
Gráfico de aperturas a las 3 galaxias
"""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F140W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F140W, interval=ZScaleInterval()))
central_aperture_F140W_472.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F140W_472.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_472_F140W, x_max_472_F140W, y_min_472_F140W, y_max_472_F140W])

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F140W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F140W, interval=ZScaleInterval()))
central_aperture_F140W_217.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F140W_217.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_217_F140W, x_max_217_F140W, y_min_217_F140W, y_max_217_F140W])

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F140W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F140W, interval=ZScaleInterval()))
central_aperture_F140W_481.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F140W_481.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_481_F140W, x_max_481_F140W, y_min_481_F140W, y_max_481_F140W])

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F140W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F140W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Fotometria de apertura con sustracción de fondo a CEERSYJ-0012159472,
CEERSYJ-9586559217 y CEERSYJ-0012959481 con una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

positions = [position_472_F140W, position_217_F140W, position_481_F140W]

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2

aperture = CircularAperture(positions, r=aperture_radius)
aperture_area = aperture.area

annulus_aperture = CircularAnnulus(
    positions, r_in=aperture_radius*2, r_out=aperture_radius*3)

aperstats = ApertureStats(data_F140W, annulus_aperture)
bkg_mean = aperstats.mean

total_bkg = bkg_mean * aperture_area

phot_table_F140W = aperture_photometry(data_F140W, aperture)
phot_bkgsub = phot_table_F140W['aperture_sum'] - total_bkg
phot_table_F140W['total_bkg'] = total_bkg
phot_table_F140W['aperture_sum_bkgsub'] = phot_bkgsub

for col in phot_table_F140W.colnames:
    phot_table_F140W[col].info.format = '%.8g'  # for consistent table output

print(phot_table_F140W)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Obtención de errores de la fotometría de apertura con
sustracción de fondo y para una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2
N_apertures = 50

# CEERSYJ-0012159472

cutout_472_F140W = data_F140W[int(np.floor(x_min_472_F140W)):int(np.ceil(
    x_max_472_F140W)), int(np.floor(y_min_472_F140W)):int(np.ceil(y_max_472_F140W))]

lower_x_472_F140W, upper_x_472_F140W = int(np.ceil(
    x_min_472_F140W + aperture_radius*3)), int(np.floor(x_max_472_F140W - aperture_radius*3))
lower_y_472_F140W, upper_y_472_F140W = int(np.ceil(
    y_min_472_F140W + aperture_radius*3)), int(np.floor(y_max_472_F140W - aperture_radius*3))

x_centers_472_F140W = [random.randrange(
    start=lower_x_472_F140W, stop=upper_x_472_F140W) for i in range(N_apertures)]
y_centers_472_F140W = [random.randrange(
    start=lower_y_472_F140W, stop=upper_y_472_F140W) for i in range(N_apertures)]

random_apertures_472_F140W = CircularAperture(
    zip(x_centers_472_F140W, y_centers_472_F140W), r=aperture_radius)
random_apertures_472_F140W_area = aperture.area

random_annulus_472_F140W = CircularAnnulus(zip(
    x_centers_472_F140W, y_centers_472_F140W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_472_F140W = ApertureStats(data_F140W, random_annulus_472_F140W)
annulus_bkg_mean_472_F140W = apertures_stats_472_F140W.mean

total_random_bkg_472_F140W = annulus_bkg_mean_472_F140W * \
    random_apertures_472_F140W_area

phot_table_472_F140W = aperture_photometry(
    data_F140W, random_apertures_472_F140W)
phot_bkgsub_472_F140W = phot_table_472_F140W['aperture_sum'] - \
    total_random_bkg_472_F140W
phot_table_472_F140W['total_bkg'] = total_random_bkg_472_F140W
phot_table_472_F140W['aperture_sum_bkgsub'] = phot_bkgsub_472_F140W

for col in phot_table_472_F140W.colnames:
    # for consistent table output
    phot_table_472_F140W[col].info.format = '%.8g'

print(phot_table_472_F140W)

fluxes_472_F140W = phot_table_472_F140W['aperture_sum_bkgsub']
print(len(fluxes_472_F140W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_472_F140W)))
fluxes_472_F140W = fluxes_472_F140W[np.where(
    fluxes_472_F140W < 1*np.std(fluxes_472_F140W))]
print(len(fluxes_472_F140W))
fluxes_472_F140W_std = np.std(fluxes_472_F140W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_472_F140W_std))

# CEERSYJ-9586559217

cutout_217_F140W = data_F140W[int(np.floor(x_min_217_F140W)):int(np.ceil(
    x_max_217_F140W)), int(np.floor(y_min_217_F140W)):int(np.ceil(y_max_217_F140W))]

lower_x_217_F140W, upper_x_217_F140W = int(np.ceil(
    x_min_217_F140W + aperture_radius*3)), int(np.floor(x_max_217_F140W - aperture_radius*3))
lower_y_217_F140W, upper_y_217_F140W = int(np.ceil(
    y_min_217_F140W + aperture_radius*3)), int(np.floor(y_max_217_F140W - aperture_radius*3))

x_centers_217_F140W = [random.randrange(
    start=lower_x_217_F140W, stop=upper_x_217_F140W) for i in range(N_apertures)]
y_centers_217_F140W = [random.randrange(
    start=lower_y_217_F140W, stop=upper_y_217_F140W) for i in range(N_apertures)]

random_apertures_217_F140W = CircularAperture(
    zip(x_centers_217_F140W, y_centers_217_F140W), r=aperture_radius)
random_apertures_217_F140W_area = aperture.area

random_annulus_217_F140W = CircularAnnulus(zip(
    x_centers_217_F140W, y_centers_217_F140W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_217_F140W = ApertureStats(data_F140W, random_annulus_217_F140W)
annulus_bkg_mean_217_F140W = apertures_stats_217_F140W.mean

total_random_bkg_217_F140W = annulus_bkg_mean_217_F140W * \
    random_apertures_217_F140W_area

phot_table_217_F140W = aperture_photometry(
    data_F140W, random_apertures_217_F140W)
phot_bkgsub_217_F140W = phot_table_217_F140W['aperture_sum'] - \
    total_random_bkg_217_F140W
phot_table_217_F140W['total_bkg'] = total_random_bkg_217_F140W
phot_table_217_F140W['aperture_sum_bkgsub'] = phot_bkgsub_217_F140W

for col in phot_table_217_F140W.colnames:
    # for consistent table output
    phot_table_217_F140W[col].info.format = '%.8g'

print(phot_table_217_F140W)

fluxes_217_F140W = phot_table_217_F140W['aperture_sum_bkgsub']
print(len(fluxes_217_F140W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_217_F140W)))
fluxes_217_F140W = fluxes_217_F140W[np.where(
    fluxes_217_F140W < 1*np.std(fluxes_217_F140W))]
print(len(fluxes_217_F140W))
fluxes_217_F140W_std = np.std(fluxes_217_F140W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_217_F140W_std))

# CEERSYJ-0012959481

cutout_481_F140W = data_F140W[int(np.floor(x_min_481_F140W)):int(np.ceil(
    x_max_481_F140W)), int(np.floor(y_min_481_F140W)):int(np.ceil(y_max_481_F140W))]

lower_x_481_F140W, upper_x_481_F140W = int(np.ceil(
    x_min_481_F140W + aperture_radius*3)), int(np.floor(x_max_481_F140W - aperture_radius*3))
lower_y_481_F140W, upper_y_481_F140W = int(np.ceil(
    y_min_481_F140W + aperture_radius*3)), int(np.floor(y_max_481_F140W - aperture_radius*3))

x_centers_481_F140W = [random.randrange(
    start=lower_x_481_F140W, stop=upper_x_481_F140W) for i in range(N_apertures)]
y_centers_481_F140W = [random.randrange(
    start=lower_y_481_F140W, stop=upper_y_481_F140W) for i in range(N_apertures)]

random_apertures_481_F140W = CircularAperture(
    zip(x_centers_481_F140W, y_centers_481_F140W), r=aperture_radius)
random_apertures_481_F140W_area = aperture.area

random_annulus_481_F140W = CircularAnnulus(zip(
    x_centers_481_F140W, y_centers_481_F140W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_481_F140W = ApertureStats(data_F140W, random_annulus_481_F140W)
annulus_bkg_mean_481_F140W = apertures_stats_481_F140W.mean

total_random_bkg_481_F140W = annulus_bkg_mean_481_F140W * \
    random_apertures_481_F140W_area

phot_table_481_F140W = aperture_photometry(
    data_F140W, random_apertures_481_F140W)
phot_bkgsub_481_F140W = phot_table_481_F140W['aperture_sum'] - \
    total_random_bkg_481_F140W
phot_table_481_F140W['total_bkg'] = total_random_bkg_481_F140W
phot_table_481_F140W['aperture_sum_bkgsub'] = phot_bkgsub_481_F140W

for col in phot_table_481_F140W.colnames:
    # for consistent table output
    phot_table_481_F140W[col].info.format = '%.8g'

print(phot_table_481_F140W)

fluxes_481_F140W = phot_table_481_F140W['aperture_sum_bkgsub']
print(len(fluxes_481_F140W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_481_F140W)))
fluxes_481_F140W = fluxes_481_F140W[np.where(
    fluxes_481_F140W < 1*np.std(fluxes_481_F140W))]
print(len(fluxes_481_F140W))
fluxes_481_F140W_std = np.std(fluxes_481_F140W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_481_F140W_std))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Gráfico de apertura de puntos azarosos cercanos a
CEERSYJ-0012159472, CEERSYJ-9586559217 y CEERSYJ-0012959481
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F140W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F140W, interval=ZScaleInterval()))
central_aperture_F140W_472.plot(color='yellow', lw=2)
random_apertures_472_F140W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlim((int(np.floor(x_min_472_F140W)), int(np.ceil(x_max_472_F140W))))
plt.ylim((int(np.floor(y_min_472_F140W)), int(np.ceil(y_max_472_F140W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F140W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F140W, interval=ZScaleInterval()))
central_aperture_F140W_217.plot(color='yellow', lw=2)
random_apertures_217_F140W.plot(color='lime', lw=2)
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlim((int(np.floor(x_min_217_F140W)), int(np.ceil(x_max_217_F140W))))
plt.ylim((int(np.floor(y_min_217_F140W)), int(np.ceil(y_max_217_F140W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F140W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F140W, interval=ZScaleInterval()))
central_aperture_F140W_481.plot(color='yellow', lw=2)
random_apertures_481_F140W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlim((int(np.floor(x_min_481_F140W)), int(np.ceil(x_max_481_F140W))))
plt.ylim((int(np.floor(y_min_481_F140W)), int(np.ceil(y_max_481_F140W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F140W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F140W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
for i, ax in enumerate(axs):
    axs[i].legend(
        ['Apertura del objeto', 'Apertura de puntos azarosos'], loc='upper right')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Lectura archivos .fits de filtro F150W
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

hdul_F150W = fits.open(
    '/home/javier/Escritorio/TTB/nircam1/hlsp_ceers_jwst_nircam_nircam1_f150w_dr0.5_i2d.fits')
hdr_F150W = hdul_F150W[0].header
data_F150W = hdul_F150W[1].data

pixel_scale = 0.03  # arcsecs/pixel


"""""""""""""""""""""""""""""""""""""""""""""
Generación de apertura y anillos de fondo
"""""""""""""""""""""""""""""""""""""""""""""

# CEERSYJ-0012159472

aperture_diameter_arcsecs_472 = 0.3  # 0.35
aperture_radius_472 = aperture_diameter_arcsecs_472 / pixel_scale / 2

xc_472_F150W, yc_472_F150W = 2563.17, 2677.76
position_472_F150W = (xc_472_F150W, yc_472_F150W)

central_aperture_F150W_472 = CircularAperture(
    position_472_F150W, r=aperture_radius_472)
annulus_aperture_F150W_472 = CircularAnnulus(
    position_472_F150W, r_in=aperture_radius_472*2, r_out=aperture_radius_472*3)

delta_arcsecs_472_F150W = 3
delta_pix_472_F150W = delta_arcsecs_472_F150W / pixel_scale
x_min_472_F150W = xc_472_F150W - delta_pix_472_F150W
x_max_472_F150W = xc_472_F150W + delta_pix_472_F150W
y_min_472_F150W = yc_472_F150W - delta_pix_472_F150W
y_max_472_F150W = yc_472_F150W + delta_pix_472_F150W

# CEERSYJ-9586559217

aperture_diameter_arcsecs_217 = 0.3  # 0.208
aperture_radius_217 = aperture_diameter_arcsecs_217 / pixel_scale / 2

xc_217_F150W, yc_217_F150W = 3708.89, 2710.52
position_217_F150W = (xc_217_F150W, yc_217_F150W)

central_aperture_F150W_217 = CircularAperture(
    position_217_F150W, r=aperture_radius_217)
annulus_aperture_F150W_217 = CircularAnnulus(
    position_217_F150W, r_in=aperture_radius_217*2, r_out=aperture_radius_217*3)

delta_arcsecs_217_F150W = 3
delta_pix_217_F150W = delta_arcsecs_217_F150W / pixel_scale
x_min_217_F150W = xc_217_F150W - delta_pix_217_F150W
x_max_217_F150W = xc_217_F150W + delta_pix_217_F150W
y_min_217_F150W = yc_217_F150W - delta_pix_217_F150W
y_max_217_F150W = yc_217_F150W + delta_pix_217_F150W

# CEERSYJ-0012959481

aperture_diameter_arcsecs_481 = 0.3  # 0.336
aperture_radius_481 = aperture_diameter_arcsecs_481 / pixel_scale / 2

xc_481_F150W, yc_481_F150W = 2526.86, 2677.60
position_481_F150W = (xc_481_F150W, yc_481_F150W)

central_aperture_F150W_481 = CircularAperture(
    position_481_F150W, r=aperture_radius_481)
annulus_aperture_F150W_481 = CircularAnnulus(
    position_481_F150W, r_in=aperture_radius_481*2, r_out=aperture_radius_481*3)

delta_arcsecs_481_F150W = 3
delta_pix_481_F150W = delta_arcsecs_481_F150W / pixel_scale
x_min_481_F150W = xc_481_F150W - delta_pix_481_F150W
x_max_481_F150W = xc_481_F150W + delta_pix_481_F150W
y_min_481_F150W = yc_481_F150W - delta_pix_481_F150W
y_max_481_F150W = yc_481_F150W + delta_pix_481_F150W


"""""""""""""""""""""""""""""""""""""""
Gráfico de aperturas a las 3 galaxias
"""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F150W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F150W, interval=ZScaleInterval()))
central_aperture_F150W_472.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F150W_472.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_472_F150W, x_max_472_F150W, y_min_472_F150W, y_max_472_F150W])

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F150W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F150W, interval=ZScaleInterval()))
central_aperture_F150W_217.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F150W_217.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_217_F150W, x_max_217_F150W, y_min_217_F150W, y_max_217_F150W])

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F150W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F150W, interval=ZScaleInterval()))
central_aperture_F150W_481.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F150W_481.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_481_F150W, x_max_481_F150W, y_min_481_F150W, y_max_481_F150W])

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F150W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F150W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Fotometria de apertura con sustracción de fondo a CEERSYJ-0012159472,
CEERSYJ-9586559217 y CEERSYJ-0012959481 con una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

positions = [position_472_F150W, position_217_F150W, position_481_F150W]

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2

aperture = CircularAperture(positions, r=aperture_radius)
aperture_area = aperture.area

annulus_aperture = CircularAnnulus(
    positions, r_in=aperture_radius*2, r_out=aperture_radius*3)

aperstats = ApertureStats(data_F150W, annulus_aperture)
bkg_mean = aperstats.mean

total_bkg = bkg_mean * aperture_area

phot_table_F150W = aperture_photometry(data_F150W, aperture)
phot_bkgsub = phot_table_F150W['aperture_sum'] - total_bkg
phot_table_F150W['total_bkg'] = total_bkg
phot_table_F150W['aperture_sum_bkgsub'] = phot_bkgsub

for col in phot_table_F150W.colnames:
    phot_table_F150W[col].info.format = '%.8g'  # for consistent table output

print(phot_table_F150W)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Obtención de errores de la fotometría de apertura con
sustracción de fondo y para una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2
N_apertures = 50

# CEERSYJ-0012159472

cutout_472_F150W = data_F150W[int(np.floor(x_min_472_F150W)):int(np.ceil(
    x_max_472_F150W)), int(np.floor(y_min_472_F150W)):int(np.ceil(y_max_472_F150W))]

lower_x_472_F150W, upper_x_472_F150W = int(np.ceil(
    x_min_472_F150W + aperture_radius*3)), int(np.floor(x_max_472_F150W - aperture_radius*3))
lower_y_472_F150W, upper_y_472_F150W = int(np.ceil(
    y_min_472_F150W + aperture_radius*3)), int(np.floor(y_max_472_F150W - aperture_radius*3))

x_centers_472_F150W = [random.randrange(
    start=lower_x_472_F150W, stop=upper_x_472_F150W) for i in range(N_apertures)]
y_centers_472_F150W = [random.randrange(
    start=lower_y_472_F150W, stop=upper_y_472_F150W) for i in range(N_apertures)]

random_apertures_472_F150W = CircularAperture(
    zip(x_centers_472_F150W, y_centers_472_F150W), r=aperture_radius)
random_apertures_472_F150W_area = aperture.area

random_annulus_472_F150W = CircularAnnulus(zip(
    x_centers_472_F150W, y_centers_472_F150W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_472_F150W = ApertureStats(data_F150W, random_annulus_472_F150W)
annulus_bkg_mean_472_F150W = apertures_stats_472_F150W.mean

total_random_bkg_472_F150W = annulus_bkg_mean_472_F150W * \
    random_apertures_472_F150W_area

phot_table_472_F150W = aperture_photometry(
    data_F150W, random_apertures_472_F150W)
phot_bkgsub_472_F150W = phot_table_472_F150W['aperture_sum'] - \
    total_random_bkg_472_F150W
phot_table_472_F150W['total_bkg'] = total_random_bkg_472_F150W
phot_table_472_F150W['aperture_sum_bkgsub'] = phot_bkgsub_472_F150W

for col in phot_table_472_F150W.colnames:
    # for consistent table output
    phot_table_472_F150W[col].info.format = '%.8g'

print(phot_table_472_F150W)

fluxes_472_F150W = phot_table_472_F150W['aperture_sum_bkgsub']
print(len(fluxes_472_F150W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_472_F150W)))
fluxes_472_F150W = fluxes_472_F150W[np.where(
    fluxes_472_F150W < 1*np.std(fluxes_472_F150W))]
print(len(fluxes_472_F150W))
fluxes_472_F150W_std = np.std(fluxes_472_F150W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_472_F150W_std))

# CEERSYJ-9586559217

cutout_217_F150W = data_F150W[int(np.floor(x_min_217_F150W)):int(np.ceil(
    x_max_217_F150W)), int(np.floor(y_min_217_F150W)):int(np.ceil(y_max_217_F150W))]

lower_x_217_F150W, upper_x_217_F150W = int(np.ceil(
    x_min_217_F150W + aperture_radius*3)), int(np.floor(x_max_217_F150W - aperture_radius*3))
lower_y_217_F150W, upper_y_217_F150W = int(np.ceil(
    y_min_217_F150W + aperture_radius*3)), int(np.floor(y_max_217_F150W - aperture_radius*3))

x_centers_217_F150W = [random.randrange(
    start=lower_x_217_F150W, stop=upper_x_217_F150W) for i in range(N_apertures)]
y_centers_217_F150W = [random.randrange(
    start=lower_y_217_F150W, stop=upper_y_217_F150W) for i in range(N_apertures)]

random_apertures_217_F150W = CircularAperture(
    zip(x_centers_217_F150W, y_centers_217_F150W), r=aperture_radius)
random_apertures_217_F150W_area = aperture.area

random_annulus_217_F150W = CircularAnnulus(zip(
    x_centers_217_F150W, y_centers_217_F150W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_217_F150W = ApertureStats(data_F150W, random_annulus_217_F150W)
annulus_bkg_mean_217_F150W = apertures_stats_217_F150W.mean

total_random_bkg_217_F150W = annulus_bkg_mean_217_F150W * \
    random_apertures_217_F150W_area

phot_table_217_F150W = aperture_photometry(
    data_F150W, random_apertures_217_F150W)
phot_bkgsub_217_F150W = phot_table_217_F150W['aperture_sum'] - \
    total_random_bkg_217_F150W
phot_table_217_F150W['total_bkg'] = total_random_bkg_217_F150W
phot_table_217_F150W['aperture_sum_bkgsub'] = phot_bkgsub_217_F150W

for col in phot_table_217_F150W.colnames:
    # for consistent table output
    phot_table_217_F150W[col].info.format = '%.8g'

print(phot_table_217_F150W)

fluxes_217_F150W = phot_table_217_F150W['aperture_sum_bkgsub']
print(len(fluxes_217_F150W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_217_F150W)))
fluxes_217_F150W = fluxes_217_F150W[np.where(
    fluxes_217_F150W < 1*np.std(fluxes_217_F150W))]
print(len(fluxes_217_F150W))
fluxes_217_F150W_std = np.std(fluxes_217_F150W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_217_F150W_std))

# CEERSYJ-0012959481

cutout_481_F150W = data_F150W[int(np.floor(x_min_481_F150W)):int(np.ceil(
    x_max_481_F150W)), int(np.floor(y_min_481_F150W)):int(np.ceil(y_max_481_F150W))]

lower_x_481_F150W, upper_x_481_F150W = int(np.ceil(
    x_min_481_F150W + aperture_radius*3)), int(np.floor(x_max_481_F150W - aperture_radius*3))
lower_y_481_F150W, upper_y_481_F150W = int(np.ceil(
    y_min_481_F150W + aperture_radius*3)), int(np.floor(y_max_481_F150W - aperture_radius*3))

x_centers_481_F150W = [random.randrange(
    start=lower_x_481_F150W, stop=upper_x_481_F150W) for i in range(N_apertures)]
y_centers_481_F150W = [random.randrange(
    start=lower_y_481_F150W, stop=upper_y_481_F150W) for i in range(N_apertures)]

random_apertures_481_F150W = CircularAperture(
    zip(x_centers_481_F150W, y_centers_481_F150W), r=aperture_radius)
random_apertures_481_F150W_area = aperture.area

random_annulus_481_F150W = CircularAnnulus(zip(
    x_centers_481_F150W, y_centers_481_F150W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_481_F150W = ApertureStats(data_F150W, random_annulus_481_F150W)
annulus_bkg_mean_481_F150W = apertures_stats_481_F150W.mean

total_random_bkg_481_F150W = annulus_bkg_mean_481_F150W * \
    random_apertures_481_F150W_area

phot_table_481_F150W = aperture_photometry(
    data_F150W, random_apertures_481_F150W)
phot_bkgsub_481_F150W = phot_table_481_F150W['aperture_sum'] - \
    total_random_bkg_481_F150W
phot_table_481_F150W['total_bkg'] = total_random_bkg_481_F150W
phot_table_481_F150W['aperture_sum_bkgsub'] = phot_bkgsub_481_F150W

for col in phot_table_481_F150W.colnames:
    # for consistent table output
    phot_table_481_F150W[col].info.format = '%.8g'

print(phot_table_481_F150W)

fluxes_481_F150W = phot_table_481_F150W['aperture_sum_bkgsub']
print(len(fluxes_481_F150W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_481_F150W)))
fluxes_481_F150W = fluxes_481_F150W[np.where(
    fluxes_481_F150W < 1*np.std(fluxes_481_F150W))]
print(len(fluxes_481_F150W))
fluxes_481_F150W_std = np.std(fluxes_481_F150W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_481_F150W_std))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Gráfico de apertura de puntos azarosos cercanos a
CEERSYJ-0012159472, CEERSYJ-9586559217 y CEERSYJ-0012959481
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F150W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F150W, interval=ZScaleInterval()))
central_aperture_F150W_472.plot(color='yellow', lw=2)
random_apertures_472_F150W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlim((int(np.floor(x_min_472_F150W)), int(np.ceil(x_max_472_F150W))))
plt.ylim((int(np.floor(y_min_472_F150W)), int(np.ceil(y_max_472_F150W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F150W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F150W, interval=ZScaleInterval()))
central_aperture_F150W_217.plot(color='yellow', lw=2)
random_apertures_217_F150W.plot(color='lime', lw=2)
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlim((int(np.floor(x_min_217_F150W)), int(np.ceil(x_max_217_F150W))))
plt.ylim((int(np.floor(y_min_217_F150W)), int(np.ceil(y_max_217_F150W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F150W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F150W, interval=ZScaleInterval()))
central_aperture_F150W_481.plot(color='yellow', lw=2)
random_apertures_481_F150W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlim((int(np.floor(x_min_481_F150W)), int(np.ceil(x_max_481_F150W))))
plt.ylim((int(np.floor(y_min_481_F150W)), int(np.ceil(y_max_481_F150W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F150W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F150W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
for i, ax in enumerate(axs):
    axs[i].legend(
        ['Apertura del objeto', 'Apertura de puntos azarosos'], loc='upper right')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Lectura archivos .fits de filtro F160W
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

hdul_F160W = fits.open(
    '/home/javier/Escritorio/TTB/nircam1/egs_all_wfc3_ir_f160w_030mas_v1.9_nircam1_mef.fits')
hdr_F160W = hdul_F160W[0].header
data_F160W = hdul_F160W[1].data

pixel_scale = 0.03  # arcsecs/pixel


"""""""""""""""""""""""""""""""""""""""""""""
Generación de apertura y anillos de fondo
"""""""""""""""""""""""""""""""""""""""""""""

# CEERSYJ-0012159472

aperture_diameter_arcsecs_472 = 0.3  # 0.35
aperture_radius_472 = aperture_diameter_arcsecs_472 / pixel_scale / 2

xc_472_F160W, yc_472_F160W = 2563.17, 2677.76
position_472_F160W = (xc_472_F160W, yc_472_F160W)

central_aperture_F160W_472 = CircularAperture(
    position_472_F160W, r=aperture_radius_472)
annulus_aperture_F160W_472 = CircularAnnulus(
    position_472_F160W, r_in=aperture_radius_472*2, r_out=aperture_radius_472*3)

delta_arcsecs_472_F160W = 3
delta_pix_472_F160W = delta_arcsecs_472_F160W / pixel_scale
x_min_472_F160W = xc_472_F160W - delta_pix_472_F160W
x_max_472_F160W = xc_472_F160W + delta_pix_472_F160W
y_min_472_F160W = yc_472_F160W - delta_pix_472_F160W
y_max_472_F160W = yc_472_F160W + delta_pix_472_F160W

# CEERSYJ-9586559217

aperture_diameter_arcsecs_217 = 0.3  # 0.208
aperture_radius_217 = aperture_diameter_arcsecs_217 / pixel_scale / 2

xc_217_F160W, yc_217_F160W = 3708.89, 2710.52
position_217_F160W = (xc_217_F160W, yc_217_F160W)

central_aperture_F160W_217 = CircularAperture(
    position_217_F160W, r=aperture_radius_217)
annulus_aperture_F160W_217 = CircularAnnulus(
    position_217_F160W, r_in=aperture_radius_217*2, r_out=aperture_radius_217*3)

delta_arcsecs_217_F160W = 3
delta_pix_217_F160W = delta_arcsecs_217_F160W / pixel_scale
x_min_217_F160W = xc_217_F160W - delta_pix_217_F160W
x_max_217_F160W = xc_217_F160W + delta_pix_217_F160W
y_min_217_F160W = yc_217_F160W - delta_pix_217_F160W
y_max_217_F160W = yc_217_F160W + delta_pix_217_F160W

# CEERSYJ-0012959481

aperture_diameter_arcsecs_481 = 0.3  # 0.336
aperture_radius_481 = aperture_diameter_arcsecs_481 / pixel_scale / 2

xc_481_F160W, yc_481_F160W = 2526.86, 2677.60
position_481_F160W = (xc_481_F160W, yc_481_F160W)

central_aperture_F160W_481 = CircularAperture(
    position_481_F160W, r=aperture_radius_481)
annulus_aperture_F160W_481 = CircularAnnulus(
    position_481_F160W, r_in=aperture_radius_481*2, r_out=aperture_radius_481*3)

delta_arcsecs_481_F160W = 3
delta_pix_481_F160W = delta_arcsecs_481_F160W / pixel_scale
x_min_481_F160W = xc_481_F160W - delta_pix_481_F160W
x_max_481_F160W = xc_481_F160W + delta_pix_481_F160W
y_min_481_F160W = yc_481_F160W - delta_pix_481_F160W
y_max_481_F160W = yc_481_F160W + delta_pix_481_F160W


"""""""""""""""""""""""""""""""""""""""
Gráfico de aperturas a las 3 galaxias
"""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F160W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F160W, interval=ZScaleInterval()))
central_aperture_F160W_472.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F160W_472.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_472_F160W, x_max_472_F160W, y_min_472_F160W, y_max_472_F160W])

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F160W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F160W, interval=ZScaleInterval()))
central_aperture_F160W_217.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F160W_217.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_217_F160W, x_max_217_F160W, y_min_217_F160W, y_max_217_F160W])

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F160W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F160W, interval=ZScaleInterval()))
central_aperture_F160W_481.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F160W_481.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_481_F160W, x_max_481_F160W, y_min_481_F160W, y_max_481_F160W])

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F160W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F160W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Fotometria de apertura con sustracción de fondo a CEERSYJ-0012159472,
CEERSYJ-9586559217 y CEERSYJ-0012959481 con una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

positions = [position_472_F160W, position_217_F160W, position_481_F160W]

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2

aperture = CircularAperture(positions, r=aperture_radius)
aperture_area = aperture.area

annulus_aperture = CircularAnnulus(
    positions, r_in=aperture_radius*2, r_out=aperture_radius*3)

aperstats = ApertureStats(data_F160W, annulus_aperture)
bkg_mean = aperstats.mean

total_bkg = bkg_mean * aperture_area

phot_table_F160W = aperture_photometry(data_F160W, aperture)
phot_bkgsub = phot_table_F160W['aperture_sum'] - total_bkg
phot_table_F160W['total_bkg'] = total_bkg
phot_table_F160W['aperture_sum_bkgsub'] = phot_bkgsub

for col in phot_table_F160W.colnames:
    phot_table_F160W[col].info.format = '%.8g'  # for consistent table output

print(phot_table_F160W)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Obtención de errores de la fotometría de apertura con
sustracción de fondo y para una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2
N_apertures = 50

# CEERSYJ-0012159472

cutout_472_F160W = data_F160W[int(np.floor(x_min_472_F160W)):int(np.ceil(
    x_max_472_F160W)), int(np.floor(y_min_472_F160W)):int(np.ceil(y_max_472_F160W))]

lower_x_472_F160W, upper_x_472_F160W = int(np.ceil(
    x_min_472_F160W + aperture_radius*3)), int(np.floor(x_max_472_F160W - aperture_radius*3))
lower_y_472_F160W, upper_y_472_F160W = int(np.ceil(
    y_min_472_F160W + aperture_radius*3)), int(np.floor(y_max_472_F160W - aperture_radius*3))

x_centers_472_F160W = [random.randrange(
    start=lower_x_472_F160W, stop=upper_x_472_F160W) for i in range(N_apertures)]
y_centers_472_F160W = [random.randrange(
    start=lower_y_472_F160W, stop=upper_y_472_F160W) for i in range(N_apertures)]

random_apertures_472_F160W = CircularAperture(
    zip(x_centers_472_F160W, y_centers_472_F160W), r=aperture_radius)
random_apertures_472_F160W_area = aperture.area

random_annulus_472_F160W = CircularAnnulus(zip(
    x_centers_472_F160W, y_centers_472_F160W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_472_F160W = ApertureStats(data_F160W, random_annulus_472_F160W)
annulus_bkg_mean_472_F160W = apertures_stats_472_F160W.mean

total_random_bkg_472_F160W = annulus_bkg_mean_472_F160W * \
    random_apertures_472_F160W_area

phot_table_472_F160W = aperture_photometry(
    data_F160W, random_apertures_472_F160W)
phot_bkgsub_472_F160W = phot_table_472_F160W['aperture_sum'] - \
    total_random_bkg_472_F160W
phot_table_472_F160W['total_bkg'] = total_random_bkg_472_F160W
phot_table_472_F160W['aperture_sum_bkgsub'] = phot_bkgsub_472_F160W

for col in phot_table_472_F160W.colnames:
    # for consistent table output
    phot_table_472_F160W[col].info.format = '%.8g'

print(phot_table_472_F160W)

fluxes_472_F160W = phot_table_472_F160W['aperture_sum_bkgsub']
print(len(fluxes_472_F160W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_472_F160W)))
fluxes_472_F160W = fluxes_472_F160W[np.where(
    fluxes_472_F160W < 1*np.std(fluxes_472_F160W))]
print(len(fluxes_472_F160W))
fluxes_472_F160W_std = np.std(fluxes_472_F160W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_472_F160W_std))

# CEERSYJ-9586559217

cutout_217_F160W = data_F160W[int(np.floor(x_min_217_F160W)):int(np.ceil(
    x_max_217_F160W)), int(np.floor(y_min_217_F160W)):int(np.ceil(y_max_217_F160W))]

lower_x_217_F160W, upper_x_217_F160W = int(np.ceil(
    x_min_217_F160W + aperture_radius*3)), int(np.floor(x_max_217_F160W - aperture_radius*3))
lower_y_217_F160W, upper_y_217_F160W = int(np.ceil(
    y_min_217_F160W + aperture_radius*3)), int(np.floor(y_max_217_F160W - aperture_radius*3))

x_centers_217_F160W = [random.randrange(
    start=lower_x_217_F160W, stop=upper_x_217_F160W) for i in range(N_apertures)]
y_centers_217_F160W = [random.randrange(
    start=lower_y_217_F160W, stop=upper_y_217_F160W) for i in range(N_apertures)]

random_apertures_217_F160W = CircularAperture(
    zip(x_centers_217_F160W, y_centers_217_F160W), r=aperture_radius)
random_apertures_217_F160W_area = aperture.area

random_annulus_217_F160W = CircularAnnulus(zip(
    x_centers_217_F160W, y_centers_217_F160W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_217_F160W = ApertureStats(data_F160W, random_annulus_217_F160W)
annulus_bkg_mean_217_F160W = apertures_stats_217_F160W.mean

total_random_bkg_217_F160W = annulus_bkg_mean_217_F160W * \
    random_apertures_217_F160W_area

phot_table_217_F160W = aperture_photometry(
    data_F160W, random_apertures_217_F160W)
phot_bkgsub_217_F160W = phot_table_217_F160W['aperture_sum'] - \
    total_random_bkg_217_F160W
phot_table_217_F160W['total_bkg'] = total_random_bkg_217_F160W
phot_table_217_F160W['aperture_sum_bkgsub'] = phot_bkgsub_217_F160W

for col in phot_table_217_F160W.colnames:
    # for consistent table output
    phot_table_217_F160W[col].info.format = '%.8g'

print(phot_table_217_F160W)

fluxes_217_F160W = phot_table_217_F160W['aperture_sum_bkgsub']
print(len(fluxes_217_F160W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_217_F160W)))
fluxes_217_F160W = fluxes_217_F160W[np.where(
    fluxes_217_F160W < 1*np.std(fluxes_217_F160W))]
print(len(fluxes_217_F160W))
fluxes_217_F160W_std = np.std(fluxes_217_F160W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_217_F160W_std))

# CEERSYJ-0012959481

cutout_481_F160W = data_F160W[int(np.floor(x_min_481_F160W)):int(np.ceil(
    x_max_481_F160W)), int(np.floor(y_min_481_F160W)):int(np.ceil(y_max_481_F160W))]

lower_x_481_F160W, upper_x_481_F160W = int(np.ceil(
    x_min_481_F160W + aperture_radius*3)), int(np.floor(x_max_481_F160W - aperture_radius*3))
lower_y_481_F160W, upper_y_481_F160W = int(np.ceil(
    y_min_481_F160W + aperture_radius*3)), int(np.floor(y_max_481_F160W - aperture_radius*3))

x_centers_481_F160W = [random.randrange(
    start=lower_x_481_F160W, stop=upper_x_481_F160W) for i in range(N_apertures)]
y_centers_481_F160W = [random.randrange(
    start=lower_y_481_F160W, stop=upper_y_481_F160W) for i in range(N_apertures)]

random_apertures_481_F160W = CircularAperture(
    zip(x_centers_481_F160W, y_centers_481_F160W), r=aperture_radius)
random_apertures_481_F160W_area = aperture.area

random_annulus_481_F160W = CircularAnnulus(zip(
    x_centers_481_F160W, y_centers_481_F160W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_481_F160W = ApertureStats(data_F160W, random_annulus_481_F160W)
annulus_bkg_mean_481_F160W = apertures_stats_481_F160W.mean

total_random_bkg_481_F160W = annulus_bkg_mean_481_F160W * \
    random_apertures_481_F160W_area

phot_table_481_F160W = aperture_photometry(
    data_F160W, random_apertures_481_F160W)
phot_bkgsub_481_F160W = phot_table_481_F160W['aperture_sum'] - \
    total_random_bkg_481_F160W
phot_table_481_F160W['total_bkg'] = total_random_bkg_481_F160W
phot_table_481_F160W['aperture_sum_bkgsub'] = phot_bkgsub_481_F160W

for col in phot_table_481_F160W.colnames:
    # for consistent table output
    phot_table_481_F160W[col].info.format = '%.8g'

print(phot_table_481_F160W)

fluxes_481_F160W = phot_table_481_F160W['aperture_sum_bkgsub']
print(len(fluxes_481_F160W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_481_F160W)))
fluxes_481_F160W = fluxes_481_F160W[np.where(
    fluxes_481_F160W < 1*np.std(fluxes_481_F160W))]
print(len(fluxes_481_F160W))
fluxes_481_F160W_std = np.std(fluxes_481_F160W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_481_F160W_std))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Gráfico de apertura de puntos azarosos cercanos a
CEERSYJ-0012159472, CEERSYJ-9586559217 y CEERSYJ-0012959481
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F160W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F160W, interval=ZScaleInterval()))
central_aperture_F160W_472.plot(color='yellow', lw=2)
random_apertures_472_F160W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlim((int(np.floor(x_min_472_F160W)), int(np.ceil(x_max_472_F160W))))
plt.ylim((int(np.floor(y_min_472_F160W)), int(np.ceil(y_max_472_F160W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F160W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F160W, interval=ZScaleInterval()))
central_aperture_F160W_217.plot(color='yellow', lw=2)
random_apertures_217_F160W.plot(color='lime', lw=2)
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlim((int(np.floor(x_min_217_F160W)), int(np.ceil(x_max_217_F160W))))
plt.ylim((int(np.floor(y_min_217_F160W)), int(np.ceil(y_max_217_F160W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F160W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F160W, interval=ZScaleInterval()))
central_aperture_F160W_481.plot(color='yellow', lw=2)
random_apertures_481_F160W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlim((int(np.floor(x_min_481_F160W)), int(np.ceil(x_max_481_F160W))))
plt.ylim((int(np.floor(y_min_481_F160W)), int(np.ceil(y_max_481_F160W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F160W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F160W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
for i, ax in enumerate(axs):
    axs[i].legend(
        ['Apertura del objeto', 'Apertura de puntos azarosos'], loc='upper right')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Lectura archivos .fits de filtro F200W
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

hdul_F200W = fits.open(
    '/home/javier/Escritorio/TTB/nircam1/hlsp_ceers_jwst_nircam_nircam1_f200w_dr0.5_i2d.fits')
hdr_F200W = hdul_F200W[0].header
data_F200W = hdul_F200W[1].data

pixel_scale = 0.03  # arcsecs/pixel


"""""""""""""""""""""""""""""""""""""""""""""
Generación de apertura y anillos de fondo
"""""""""""""""""""""""""""""""""""""""""""""

# CEERSYJ-0012159472

aperture_diameter_arcsecs_472 = 0.3  # 0.35
aperture_radius_472 = aperture_diameter_arcsecs_472 / pixel_scale / 2

xc_472_F200W, yc_472_F200W = 2563.17, 2677.76
position_472_F200W = (xc_472_F200W, yc_472_F200W)

central_aperture_F200W_472 = CircularAperture(
    position_472_F200W, r=aperture_radius_472)
annulus_aperture_F200W_472 = CircularAnnulus(
    position_472_F200W, r_in=aperture_radius_472*2, r_out=aperture_radius_472*3)

delta_arcsecs_472_F200W = 3
delta_pix_472_F200W = delta_arcsecs_472_F200W / pixel_scale
x_min_472_F200W = xc_472_F200W - delta_pix_472_F200W
x_max_472_F200W = xc_472_F200W + delta_pix_472_F200W
y_min_472_F200W = yc_472_F200W - delta_pix_472_F200W
y_max_472_F200W = yc_472_F200W + delta_pix_472_F200W

# CEERSYJ-9586559217

aperture_diameter_arcsecs_217 = 0.3  # 0.208
aperture_radius_217 = aperture_diameter_arcsecs_217 / pixel_scale / 2

xc_217_F200W, yc_217_F200W = 3708.89, 2710.52
position_217_F200W = (xc_217_F200W, yc_217_F200W)

central_aperture_F200W_217 = CircularAperture(
    position_217_F200W, r=aperture_radius_217)
annulus_aperture_F200W_217 = CircularAnnulus(
    position_217_F200W, r_in=aperture_radius_217*2, r_out=aperture_radius_217*3)

delta_arcsecs_217_F200W = 3
delta_pix_217_F200W = delta_arcsecs_217_F200W / pixel_scale
x_min_217_F200W = xc_217_F200W - delta_pix_217_F200W
x_max_217_F200W = xc_217_F200W + delta_pix_217_F200W
y_min_217_F200W = yc_217_F200W - delta_pix_217_F200W
y_max_217_F200W = yc_217_F200W + delta_pix_217_F200W

# CEERSYJ-0012959481

aperture_diameter_arcsecs_481 = 0.3  # 0.336
aperture_radius_481 = aperture_diameter_arcsecs_481 / pixel_scale / 2

xc_481_F200W, yc_481_F200W = 2526.86, 2677.60
position_481_F200W = (xc_481_F200W, yc_481_F200W)

central_aperture_F200W_481 = CircularAperture(
    position_481_F200W, r=aperture_radius_481)
annulus_aperture_F200W_481 = CircularAnnulus(
    position_481_F200W, r_in=aperture_radius_481*2, r_out=aperture_radius_481*3)

delta_arcsecs_481_F200W = 3
delta_pix_481_F200W = delta_arcsecs_481_F200W / pixel_scale
x_min_481_F200W = xc_481_F200W - delta_pix_481_F200W
x_max_481_F200W = xc_481_F200W + delta_pix_481_F200W
y_min_481_F200W = yc_481_F200W - delta_pix_481_F200W
y_max_481_F200W = yc_481_F200W + delta_pix_481_F200W


"""""""""""""""""""""""""""""""""""""""
Gráfico de aperturas a las 3 galaxias
"""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F200W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F200W, interval=ZScaleInterval()))
central_aperture_F200W_472.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F200W_472.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_472_F200W, x_max_472_F200W, y_min_472_F200W, y_max_472_F200W])

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F200W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F200W, interval=ZScaleInterval()))
central_aperture_F200W_217.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F200W_217.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_217_F200W, x_max_217_F200W, y_min_217_F200W, y_max_217_F200W])

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F200W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F200W, interval=ZScaleInterval()))
central_aperture_F200W_481.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F200W_481.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_481_F200W, x_max_481_F200W, y_min_481_F200W, y_max_481_F200W])

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F200W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F200W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Fotometria de apertura con sustracción de fondo a CEERSYJ-0012159472,
CEERSYJ-9586559217 y CEERSYJ-0012959481 con una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

positions = [position_472_F200W, position_217_F200W, position_481_F200W]

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2

aperture = CircularAperture(positions, r=aperture_radius)
aperture_area = aperture.area

annulus_aperture = CircularAnnulus(
    positions, r_in=aperture_radius*2, r_out=aperture_radius*3)

aperstats = ApertureStats(data_F200W, annulus_aperture)
bkg_mean = aperstats.mean

total_bkg = bkg_mean * aperture_area

phot_table_F200W = aperture_photometry(data_F200W, aperture)
phot_bkgsub = phot_table_F200W['aperture_sum'] - total_bkg
phot_table_F200W['total_bkg'] = total_bkg
phot_table_F200W['aperture_sum_bkgsub'] = phot_bkgsub

for col in phot_table_F200W.colnames:
    phot_table_F200W[col].info.format = '%.8g'  # for consistent table output

print(phot_table_F200W)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Obtención de errores de la fotometría de apertura con
sustracción de fondo y para una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2
N_apertures = 50

# CEERSYJ-0012159472

cutout_472_F200W = data_F200W[int(np.floor(x_min_472_F200W)):int(np.ceil(
    x_max_472_F200W)), int(np.floor(y_min_472_F200W)):int(np.ceil(y_max_472_F200W))]

lower_x_472_F200W, upper_x_472_F200W = int(np.ceil(
    x_min_472_F200W + aperture_radius*3)), int(np.floor(x_max_472_F200W - aperture_radius*3))
lower_y_472_F200W, upper_y_472_F200W = int(np.ceil(
    y_min_472_F200W + aperture_radius*3)), int(np.floor(y_max_472_F200W - aperture_radius*3))

x_centers_472_F200W = [random.randrange(
    start=lower_x_472_F200W, stop=upper_x_472_F200W) for i in range(N_apertures)]
y_centers_472_F200W = [random.randrange(
    start=lower_y_472_F200W, stop=upper_y_472_F200W) for i in range(N_apertures)]

random_apertures_472_F200W = CircularAperture(
    zip(x_centers_472_F200W, y_centers_472_F200W), r=aperture_radius)
random_apertures_472_F200W_area = aperture.area

random_annulus_472_F200W = CircularAnnulus(zip(
    x_centers_472_F200W, y_centers_472_F200W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_472_F200W = ApertureStats(data_F200W, random_annulus_472_F200W)
annulus_bkg_mean_472_F200W = apertures_stats_472_F200W.mean

total_random_bkg_472_F200W = annulus_bkg_mean_472_F200W * \
    random_apertures_472_F200W_area

phot_table_472_F200W = aperture_photometry(
    data_F200W, random_apertures_472_F200W)
phot_bkgsub_472_F200W = phot_table_472_F200W['aperture_sum'] - \
    total_random_bkg_472_F200W
phot_table_472_F200W['total_bkg'] = total_random_bkg_472_F200W
phot_table_472_F200W['aperture_sum_bkgsub'] = phot_bkgsub_472_F200W

for col in phot_table_472_F200W.colnames:
    # for consistent table output
    phot_table_472_F200W[col].info.format = '%.8g'

print(phot_table_472_F200W)

fluxes_472_F200W = phot_table_472_F200W['aperture_sum_bkgsub']
print(len(fluxes_472_F200W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_472_F200W)))
fluxes_472_F200W = fluxes_472_F200W[np.where(
    fluxes_472_F200W < 1*np.std(fluxes_472_F200W))]
print(len(fluxes_472_F200W))
fluxes_472_F200W_std = np.std(fluxes_472_F200W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_472_F200W_std))

# CEERSYJ-9586559217

cutout_217_F200W = data_F200W[int(np.floor(x_min_217_F200W)):int(np.ceil(
    x_max_217_F200W)), int(np.floor(y_min_217_F200W)):int(np.ceil(y_max_217_F200W))]

lower_x_217_F200W, upper_x_217_F200W = int(np.ceil(
    x_min_217_F200W + aperture_radius*3)), int(np.floor(x_max_217_F200W - aperture_radius*3))
lower_y_217_F200W, upper_y_217_F200W = int(np.ceil(
    y_min_217_F200W + aperture_radius*3)), int(np.floor(y_max_217_F200W - aperture_radius*3))

x_centers_217_F200W = [random.randrange(
    start=lower_x_217_F200W, stop=upper_x_217_F200W) for i in range(N_apertures)]
y_centers_217_F200W = [random.randrange(
    start=lower_y_217_F200W, stop=upper_y_217_F200W) for i in range(N_apertures)]

random_apertures_217_F200W = CircularAperture(
    zip(x_centers_217_F200W, y_centers_217_F200W), r=aperture_radius)
random_apertures_217_F200W_area = aperture.area

random_annulus_217_F200W = CircularAnnulus(zip(
    x_centers_217_F200W, y_centers_217_F200W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_217_F200W = ApertureStats(data_F200W, random_annulus_217_F200W)
annulus_bkg_mean_217_F200W = apertures_stats_217_F200W.mean

total_random_bkg_217_F200W = annulus_bkg_mean_217_F200W * \
    random_apertures_217_F200W_area

phot_table_217_F200W = aperture_photometry(
    data_F200W, random_apertures_217_F200W)
phot_bkgsub_217_F200W = phot_table_217_F200W['aperture_sum'] - \
    total_random_bkg_217_F200W
phot_table_217_F200W['total_bkg'] = total_random_bkg_217_F200W
phot_table_217_F200W['aperture_sum_bkgsub'] = phot_bkgsub_217_F200W

for col in phot_table_217_F200W.colnames:
    # for consistent table output
    phot_table_217_F200W[col].info.format = '%.8g'

print(phot_table_217_F200W)

fluxes_217_F200W = phot_table_217_F200W['aperture_sum_bkgsub']
print(len(fluxes_217_F200W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_217_F200W)))
fluxes_217_F200W = fluxes_217_F200W[np.where(
    fluxes_217_F200W < 1*np.std(fluxes_217_F200W))]
print(len(fluxes_217_F200W))
fluxes_217_F200W_std = np.std(fluxes_217_F200W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_217_F200W_std))

# CEERSYJ-0012959481

cutout_481_F200W = data_F200W[int(np.floor(x_min_481_F200W)):int(np.ceil(
    x_max_481_F200W)), int(np.floor(y_min_481_F200W)):int(np.ceil(y_max_481_F200W))]

lower_x_481_F200W, upper_x_481_F200W = int(np.ceil(
    x_min_481_F200W + aperture_radius*3)), int(np.floor(x_max_481_F200W - aperture_radius*3))
lower_y_481_F200W, upper_y_481_F200W = int(np.ceil(
    y_min_481_F200W + aperture_radius*3)), int(np.floor(y_max_481_F200W - aperture_radius*3))

x_centers_481_F200W = [random.randrange(
    start=lower_x_481_F200W, stop=upper_x_481_F200W) for i in range(N_apertures)]
y_centers_481_F200W = [random.randrange(
    start=lower_y_481_F200W, stop=upper_y_481_F200W) for i in range(N_apertures)]

random_apertures_481_F200W = CircularAperture(
    zip(x_centers_481_F200W, y_centers_481_F200W), r=aperture_radius)
random_apertures_481_F200W_area = aperture.area

random_annulus_481_F200W = CircularAnnulus(zip(
    x_centers_481_F200W, y_centers_481_F200W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_481_F200W = ApertureStats(data_F200W, random_annulus_481_F200W)
annulus_bkg_mean_481_F200W = apertures_stats_481_F200W.mean

total_random_bkg_481_F200W = annulus_bkg_mean_481_F200W * \
    random_apertures_481_F200W_area

phot_table_481_F200W = aperture_photometry(
    data_F200W, random_apertures_481_F200W)
phot_bkgsub_481_F200W = phot_table_481_F200W['aperture_sum'] - \
    total_random_bkg_481_F200W
phot_table_481_F200W['total_bkg'] = total_random_bkg_481_F200W
phot_table_481_F200W['aperture_sum_bkgsub'] = phot_bkgsub_481_F200W

for col in phot_table_481_F200W.colnames:
    # for consistent table output
    phot_table_481_F200W[col].info.format = '%.8g'

print(phot_table_481_F200W)

fluxes_481_F200W = phot_table_481_F200W['aperture_sum_bkgsub']
print(len(fluxes_481_F200W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_481_F200W)))
fluxes_481_F200W = fluxes_481_F200W[np.where(
    fluxes_481_F200W < 1*np.std(fluxes_481_F200W))]
print(len(fluxes_481_F200W))
fluxes_481_F200W_std = np.std(fluxes_481_F200W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_481_F200W_std))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Gráfico de apertura de puntos azarosos cercanos a
CEERSYJ-0012159472, CEERSYJ-9586559217 y CEERSYJ-0012959481
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F200W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F200W, interval=ZScaleInterval()))
central_aperture_F200W_472.plot(color='yellow', lw=2)
random_apertures_472_F200W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlim((int(np.floor(x_min_472_F200W)), int(np.ceil(x_max_472_F200W))))
plt.ylim((int(np.floor(y_min_472_F200W)), int(np.ceil(y_max_472_F200W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F200W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F200W, interval=ZScaleInterval()))
central_aperture_F200W_217.plot(color='yellow', lw=2)
random_apertures_217_F200W.plot(color='lime', lw=2)
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlim((int(np.floor(x_min_217_F200W)), int(np.ceil(x_max_217_F200W))))
plt.ylim((int(np.floor(y_min_217_F200W)), int(np.ceil(y_max_217_F200W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F200W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F200W, interval=ZScaleInterval()))
central_aperture_F200W_481.plot(color='yellow', lw=2)
random_apertures_481_F200W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlim((int(np.floor(x_min_481_F200W)), int(np.ceil(x_max_481_F200W))))
plt.ylim((int(np.floor(y_min_481_F200W)), int(np.ceil(y_max_481_F200W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F200W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F200W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
for i, ax in enumerate(axs):
    axs[i].legend(
        ['Apertura del objeto', 'Apertura de puntos azarosos'], loc='upper right')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Lectura archivos .fits de filtro F277W
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

hdul_F277W = fits.open(
    '/home/javier/Escritorio/TTB/nircam1/hlsp_ceers_jwst_nircam_nircam1_f277w_dr0.5_i2d.fits')
hdr_F277W = hdul_F277W[0].header
data_F277W = hdul_F277W[1].data

pixel_scale = 0.03  # arcsecs/pixel


"""""""""""""""""""""""""""""""""""""""""""""
Generación de apertura y anillos de fondo
"""""""""""""""""""""""""""""""""""""""""""""

# CEERSYJ-0012159472

aperture_diameter_arcsecs_472 = 0.3  # 0.35
aperture_radius_472 = aperture_diameter_arcsecs_472 / pixel_scale / 2

xc_472_F277W, yc_472_F277W = 2563.17, 2677.76
position_472_F277W = (xc_472_F277W, yc_472_F277W)

central_aperture_F277W_472 = CircularAperture(
    position_472_F277W, r=aperture_radius_472)
annulus_aperture_F277W_472 = CircularAnnulus(
    position_472_F277W, r_in=aperture_radius_472*2, r_out=aperture_radius_472*3)

delta_arcsecs_472_F277W = 3
delta_pix_472_F277W = delta_arcsecs_472_F277W / pixel_scale
x_min_472_F277W = xc_472_F277W - delta_pix_472_F277W
x_max_472_F277W = xc_472_F277W + delta_pix_472_F277W
y_min_472_F277W = yc_472_F277W - delta_pix_472_F277W
y_max_472_F277W = yc_472_F277W + delta_pix_472_F277W

# CEERSYJ-9586559217

aperture_diameter_arcsecs_217 = 0.3  # 0.208
aperture_radius_217 = aperture_diameter_arcsecs_217 / pixel_scale / 2

xc_217_F277W, yc_217_F277W = 3708.89, 2710.52
position_217_F277W = (xc_217_F277W, yc_217_F277W)

central_aperture_F277W_217 = CircularAperture(
    position_217_F277W, r=aperture_radius_217)
annulus_aperture_F277W_217 = CircularAnnulus(
    position_217_F277W, r_in=aperture_radius_217*2, r_out=aperture_radius_217*3)

delta_arcsecs_217_F277W = 3
delta_pix_217_F277W = delta_arcsecs_217_F277W / pixel_scale
x_min_217_F277W = xc_217_F277W - delta_pix_217_F277W
x_max_217_F277W = xc_217_F277W + delta_pix_217_F277W
y_min_217_F277W = yc_217_F277W - delta_pix_217_F277W
y_max_217_F277W = yc_217_F277W + delta_pix_217_F277W

# CEERSYJ-0012959481

aperture_diameter_arcsecs_481 = 0.3  # 0.336
aperture_radius_481 = aperture_diameter_arcsecs_481 / pixel_scale / 2

xc_481_F277W, yc_481_F277W = 2526.86, 2677.60
position_481_F277W = (xc_481_F277W, yc_481_F277W)

central_aperture_F277W_481 = CircularAperture(
    position_481_F277W, r=aperture_radius_481)
annulus_aperture_F277W_481 = CircularAnnulus(
    position_481_F277W, r_in=aperture_radius_481*2, r_out=aperture_radius_481*3)

delta_arcsecs_481_F277W = 3
delta_pix_481_F277W = delta_arcsecs_481_F277W / pixel_scale
x_min_481_F277W = xc_481_F277W - delta_pix_481_F277W
x_max_481_F277W = xc_481_F277W + delta_pix_481_F277W
y_min_481_F277W = yc_481_F277W - delta_pix_481_F277W
y_max_481_F277W = yc_481_F277W + delta_pix_481_F277W


"""""""""""""""""""""""""""""""""""""""
Gráfico de aperturas a las 3 galaxias
"""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F277W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F277W, interval=ZScaleInterval()))
central_aperture_F277W_472.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F277W_472.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_472_F277W, x_max_472_F277W, y_min_472_F277W, y_max_472_F277W])

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F277W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F277W, interval=ZScaleInterval()))
central_aperture_F277W_217.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F277W_217.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_217_F277W, x_max_217_F277W, y_min_217_F277W, y_max_217_F277W])

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F277W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F277W, interval=ZScaleInterval()))
central_aperture_F277W_481.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F277W_481.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_481_F277W, x_max_481_F277W, y_min_481_F277W, y_max_481_F277W])

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F277W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F277W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Fotometria de apertura con sustracción de fondo a CEERSYJ-0012159472,
CEERSYJ-9586559217 y CEERSYJ-0012959481 con una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

positions = [position_472_F277W, position_217_F277W, position_481_F277W]

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2

aperture = CircularAperture(positions, r=aperture_radius)
aperture_area = aperture.area

annulus_aperture = CircularAnnulus(
    positions, r_in=aperture_radius*2, r_out=aperture_radius*3)

aperstats = ApertureStats(data_F277W, annulus_aperture)
bkg_mean = aperstats.mean

total_bkg = bkg_mean * aperture_area

phot_table_F277W = aperture_photometry(data_F277W, aperture)
phot_bkgsub = phot_table_F277W['aperture_sum'] - total_bkg
phot_table_F277W['total_bkg'] = total_bkg
phot_table_F277W['aperture_sum_bkgsub'] = phot_bkgsub

for col in phot_table_F277W.colnames:
    phot_table_F277W[col].info.format = '%.8g'  # for consistent table output

print(phot_table_F277W)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Obtención de errores de la fotometría de apertura con
sustracción de fondo y para una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2
N_apertures = 50

# CEERSYJ-0012159472

cutout_472_F277W = data_F277W[int(np.floor(x_min_472_F277W)):int(np.ceil(
    x_max_472_F277W)), int(np.floor(y_min_472_F277W)):int(np.ceil(y_max_472_F277W))]

lower_x_472_F277W, upper_x_472_F277W = int(np.ceil(
    x_min_472_F277W + aperture_radius*3)), int(np.floor(x_max_472_F277W - aperture_radius*3))
lower_y_472_F277W, upper_y_472_F277W = int(np.ceil(
    y_min_472_F277W + aperture_radius*3)), int(np.floor(y_max_472_F277W - aperture_radius*3))

x_centers_472_F277W = [random.randrange(
    start=lower_x_472_F277W, stop=upper_x_472_F277W) for i in range(N_apertures)]
y_centers_472_F277W = [random.randrange(
    start=lower_y_472_F277W, stop=upper_y_472_F277W) for i in range(N_apertures)]

random_apertures_472_F277W = CircularAperture(
    zip(x_centers_472_F277W, y_centers_472_F277W), r=aperture_radius)
random_apertures_472_F277W_area = aperture.area

random_annulus_472_F277W = CircularAnnulus(zip(
    x_centers_472_F277W, y_centers_472_F277W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_472_F277W = ApertureStats(data_F277W, random_annulus_472_F277W)
annulus_bkg_mean_472_F277W = apertures_stats_472_F277W.mean

total_random_bkg_472_F277W = annulus_bkg_mean_472_F277W * \
    random_apertures_472_F277W_area

phot_table_472_F277W = aperture_photometry(
    data_F277W, random_apertures_472_F277W)
phot_bkgsub_472_F277W = phot_table_472_F277W['aperture_sum'] - \
    total_random_bkg_472_F277W
phot_table_472_F277W['total_bkg'] = total_random_bkg_472_F277W
phot_table_472_F277W['aperture_sum_bkgsub'] = phot_bkgsub_472_F277W

for col in phot_table_472_F277W.colnames:
    # for consistent table output
    phot_table_472_F277W[col].info.format = '%.8g'

print(phot_table_472_F277W)

fluxes_472_F277W = phot_table_472_F277W['aperture_sum_bkgsub']
print(len(fluxes_472_F277W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_472_F277W)))
fluxes_472_F277W = fluxes_472_F277W[np.where(
    fluxes_472_F277W < 1*np.std(fluxes_472_F277W))]
print(len(fluxes_472_F277W))
fluxes_472_F277W_std = np.std(fluxes_472_F277W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_472_F277W_std))

# CEERSYJ-9586559217

cutout_217_F277W = data_F277W[int(np.floor(x_min_217_F277W)):int(np.ceil(
    x_max_217_F277W)), int(np.floor(y_min_217_F277W)):int(np.ceil(y_max_217_F277W))]

lower_x_217_F277W, upper_x_217_F277W = int(np.ceil(
    x_min_217_F277W + aperture_radius*3)), int(np.floor(x_max_217_F277W - aperture_radius*3))
lower_y_217_F277W, upper_y_217_F277W = int(np.ceil(
    y_min_217_F277W + aperture_radius*3)), int(np.floor(y_max_217_F277W - aperture_radius*3))

x_centers_217_F277W = [random.randrange(
    start=lower_x_217_F277W, stop=upper_x_217_F277W) for i in range(N_apertures)]
y_centers_217_F277W = [random.randrange(
    start=lower_y_217_F277W, stop=upper_y_217_F277W) for i in range(N_apertures)]

random_apertures_217_F277W = CircularAperture(
    zip(x_centers_217_F277W, y_centers_217_F277W), r=aperture_radius)
random_apertures_217_F277W_area = aperture.area

random_annulus_217_F277W = CircularAnnulus(zip(
    x_centers_217_F277W, y_centers_217_F277W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_217_F277W = ApertureStats(data_F277W, random_annulus_217_F277W)
annulus_bkg_mean_217_F277W = apertures_stats_217_F277W.mean

total_random_bkg_217_F277W = annulus_bkg_mean_217_F277W * \
    random_apertures_217_F277W_area

phot_table_217_F277W = aperture_photometry(
    data_F277W, random_apertures_217_F277W)
phot_bkgsub_217_F277W = phot_table_217_F277W['aperture_sum'] - \
    total_random_bkg_217_F277W
phot_table_217_F277W['total_bkg'] = total_random_bkg_217_F277W
phot_table_217_F277W['aperture_sum_bkgsub'] = phot_bkgsub_217_F277W

for col in phot_table_217_F277W.colnames:
    # for consistent table output
    phot_table_217_F277W[col].info.format = '%.8g'

print(phot_table_217_F277W)

fluxes_217_F277W = phot_table_217_F277W['aperture_sum_bkgsub']
print(len(fluxes_217_F277W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_217_F277W)))
fluxes_217_F277W = fluxes_217_F277W[np.where(
    fluxes_217_F277W < 1*np.std(fluxes_217_F277W))]
print(len(fluxes_217_F277W))
fluxes_217_F277W_std = np.std(fluxes_217_F277W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_217_F277W_std))

# CEERSYJ-0012959481

cutout_481_F277W = data_F277W[int(np.floor(x_min_481_F277W)):int(np.ceil(
    x_max_481_F277W)), int(np.floor(y_min_481_F277W)):int(np.ceil(y_max_481_F277W))]

lower_x_481_F277W, upper_x_481_F277W = int(np.ceil(
    x_min_481_F277W + aperture_radius*3)), int(np.floor(x_max_481_F277W - aperture_radius*3))
lower_y_481_F277W, upper_y_481_F277W = int(np.ceil(
    y_min_481_F277W + aperture_radius*3)), int(np.floor(y_max_481_F277W - aperture_radius*3))

x_centers_481_F277W = [random.randrange(
    start=lower_x_481_F277W, stop=upper_x_481_F277W) for i in range(N_apertures)]
y_centers_481_F277W = [random.randrange(
    start=lower_y_481_F277W, stop=upper_y_481_F277W) for i in range(N_apertures)]

random_apertures_481_F277W = CircularAperture(
    zip(x_centers_481_F277W, y_centers_481_F277W), r=aperture_radius)
random_apertures_481_F277W_area = aperture.area

random_annulus_481_F277W = CircularAnnulus(zip(
    x_centers_481_F277W, y_centers_481_F277W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_481_F277W = ApertureStats(data_F277W, random_annulus_481_F277W)
annulus_bkg_mean_481_F277W = apertures_stats_481_F277W.mean

total_random_bkg_481_F277W = annulus_bkg_mean_481_F277W * \
    random_apertures_481_F277W_area

phot_table_481_F277W = aperture_photometry(
    data_F277W, random_apertures_481_F277W)
phot_bkgsub_481_F277W = phot_table_481_F277W['aperture_sum'] - \
    total_random_bkg_481_F277W
phot_table_481_F277W['total_bkg'] = total_random_bkg_481_F277W
phot_table_481_F277W['aperture_sum_bkgsub'] = phot_bkgsub_481_F277W

for col in phot_table_481_F277W.colnames:
    # for consistent table output
    phot_table_481_F277W[col].info.format = '%.8g'

print(phot_table_481_F277W)

fluxes_481_F277W = phot_table_481_F277W['aperture_sum_bkgsub']
print(len(fluxes_481_F277W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_481_F277W)))
fluxes_481_F277W = fluxes_481_F277W[np.where(
    fluxes_481_F277W < 1*np.std(fluxes_481_F277W))]
print(len(fluxes_481_F277W))
fluxes_481_F277W_std = np.std(fluxes_481_F277W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_481_F277W_std))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Gráfico de apertura de puntos azarosos cercanos a
CEERSYJ-0012159472, CEERSYJ-9586559217 y CEERSYJ-0012959481
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F277W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F277W, interval=ZScaleInterval()))
central_aperture_F277W_472.plot(color='yellow', lw=2)
random_apertures_472_F277W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlim((int(np.floor(x_min_472_F277W)), int(np.ceil(x_max_472_F277W))))
plt.ylim((int(np.floor(y_min_472_F277W)), int(np.ceil(y_max_472_F277W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F277W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F277W, interval=ZScaleInterval()))
central_aperture_F277W_217.plot(color='yellow', lw=2)
random_apertures_217_F277W.plot(color='lime', lw=2)
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlim((int(np.floor(x_min_217_F277W)), int(np.ceil(x_max_217_F277W))))
plt.ylim((int(np.floor(y_min_217_F277W)), int(np.ceil(y_max_217_F277W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F277W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F277W, interval=ZScaleInterval()))
central_aperture_F277W_481.plot(color='yellow', lw=2)
random_apertures_481_F277W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlim((int(np.floor(x_min_481_F277W)), int(np.ceil(x_max_481_F277W))))
plt.ylim((int(np.floor(y_min_481_F277W)), int(np.ceil(y_max_481_F277W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F277W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F277W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
for i, ax in enumerate(axs):
    axs[i].legend(
        ['Apertura del objeto', 'Apertura de puntos azarosos'], loc='upper right')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Lectura archivos .fits de filtro F356W
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

hdul_F356W = fits.open(
    '/home/javier/Escritorio/TTB/nircam1/hlsp_ceers_jwst_nircam_nircam1_f356w_dr0.5_i2d.fits')
hdr_F356W = hdul_F356W[0].header
data_F356W = hdul_F356W[1].data

pixel_scale = 0.03  # arcsecs/pixel


"""""""""""""""""""""""""""""""""""""""""""""
Generación de apertura y anillos de fondo
"""""""""""""""""""""""""""""""""""""""""""""

# CEERSYJ-0012159472

aperture_diameter_arcsecs_472 = 0.3  # 0.35
aperture_radius_472 = aperture_diameter_arcsecs_472 / pixel_scale / 2

xc_472_F356W, yc_472_F356W = 2563.17, 2677.76
position_472_F356W = (xc_472_F356W, yc_472_F356W)

central_aperture_F356W_472 = CircularAperture(
    position_472_F356W, r=aperture_radius_472)
annulus_aperture_F356W_472 = CircularAnnulus(
    position_472_F356W, r_in=aperture_radius_472*2, r_out=aperture_radius_472*3)

delta_arcsecs_472_F356W = 3
delta_pix_472_F356W = delta_arcsecs_472_F356W / pixel_scale
x_min_472_F356W = xc_472_F356W - delta_pix_472_F356W
x_max_472_F356W = xc_472_F356W + delta_pix_472_F356W
y_min_472_F356W = yc_472_F356W - delta_pix_472_F356W
y_max_472_F356W = yc_472_F356W + delta_pix_472_F356W

# CEERSYJ-9586559217

aperture_diameter_arcsecs_217 = 0.3  # 0.208
aperture_radius_217 = aperture_diameter_arcsecs_217 / pixel_scale / 2

xc_217_F356W, yc_217_F356W = 3708.89, 2710.52
position_217_F356W = (xc_217_F356W, yc_217_F356W)

central_aperture_F356W_217 = CircularAperture(
    position_217_F356W, r=aperture_radius_217)
annulus_aperture_F356W_217 = CircularAnnulus(
    position_217_F356W, r_in=aperture_radius_217*2, r_out=aperture_radius_217*3)

delta_arcsecs_217_F356W = 3
delta_pix_217_F356W = delta_arcsecs_217_F356W / pixel_scale
x_min_217_F356W = xc_217_F356W - delta_pix_217_F356W
x_max_217_F356W = xc_217_F356W + delta_pix_217_F356W
y_min_217_F356W = yc_217_F356W - delta_pix_217_F356W
y_max_217_F356W = yc_217_F356W + delta_pix_217_F356W

# CEERSYJ-0012959481

aperture_diameter_arcsecs_481 = 0.3  # 0.336
aperture_radius_481 = aperture_diameter_arcsecs_481 / pixel_scale / 2

xc_481_F356W, yc_481_F356W = 2526.86, 2677.60
position_481_F356W = (xc_481_F356W, yc_481_F356W)

central_aperture_F356W_481 = CircularAperture(
    position_481_F356W, r=aperture_radius_481)
annulus_aperture_F356W_481 = CircularAnnulus(
    position_481_F356W, r_in=aperture_radius_481*2, r_out=aperture_radius_481*3)

delta_arcsecs_481_F356W = 3
delta_pix_481_F356W = delta_arcsecs_481_F356W / pixel_scale
x_min_481_F356W = xc_481_F356W - delta_pix_481_F356W
x_max_481_F356W = xc_481_F356W + delta_pix_481_F356W
y_min_481_F356W = yc_481_F356W - delta_pix_481_F356W
y_max_481_F356W = yc_481_F356W + delta_pix_481_F356W


"""""""""""""""""""""""""""""""""""""""
Gráfico de aperturas a las 3 galaxias
"""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F356W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F356W, interval=ZScaleInterval()))
central_aperture_F356W_472.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F356W_472.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_472_F356W, x_max_472_F356W, y_min_472_F356W, y_max_472_F356W])

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F356W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F356W, interval=ZScaleInterval()))
central_aperture_F356W_217.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F356W_217.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_217_F356W, x_max_217_F356W, y_min_217_F356W, y_max_217_F356W])

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F356W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F356W, interval=ZScaleInterval()))
central_aperture_F356W_481.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F356W_481.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_481_F356W, x_max_481_F356W, y_min_481_F356W, y_max_481_F356W])

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F356W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F356W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Fotometria de apertura con sustracción de fondo a CEERSYJ-0012159472,
CEERSYJ-9586559217 y CEERSYJ-0012959481 con una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

positions = [position_472_F356W, position_217_F356W, position_481_F356W]

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2

aperture = CircularAperture(positions, r=aperture_radius)
aperture_area = aperture.area

annulus_aperture = CircularAnnulus(
    positions, r_in=aperture_radius*2, r_out=aperture_radius*3)

aperstats = ApertureStats(data_F356W, annulus_aperture)
bkg_mean = aperstats.mean

total_bkg = bkg_mean * aperture_area

phot_table_F356W = aperture_photometry(data_F356W, aperture)
phot_bkgsub = phot_table_F356W['aperture_sum'] - total_bkg
phot_table_F356W['total_bkg'] = total_bkg
phot_table_F356W['aperture_sum_bkgsub'] = phot_bkgsub

for col in phot_table_F356W.colnames:
    phot_table_F356W[col].info.format = '%.8g'  # for consistent table output

print(phot_table_F356W)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Obtención de errores de la fotometría de apertura con
sustracción de fondo y para una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2
N_apertures = 50

# CEERSYJ-0012159472

cutout_472_F356W = data_F356W[int(np.floor(x_min_472_F356W)):int(np.ceil(
    x_max_472_F356W)), int(np.floor(y_min_472_F356W)):int(np.ceil(y_max_472_F356W))]

lower_x_472_F356W, upper_x_472_F356W = int(np.ceil(
    x_min_472_F356W + aperture_radius*3)), int(np.floor(x_max_472_F356W - aperture_radius*3))
lower_y_472_F356W, upper_y_472_F356W = int(np.ceil(
    y_min_472_F356W + aperture_radius*3)), int(np.floor(y_max_472_F356W - aperture_radius*3))

x_centers_472_F356W = [random.randrange(
    start=lower_x_472_F356W, stop=upper_x_472_F356W) for i in range(N_apertures)]
y_centers_472_F356W = [random.randrange(
    start=lower_y_472_F356W, stop=upper_y_472_F356W) for i in range(N_apertures)]

random_apertures_472_F356W = CircularAperture(
    zip(x_centers_472_F356W, y_centers_472_F356W), r=aperture_radius)
random_apertures_472_F356W_area = aperture.area

random_annulus_472_F356W = CircularAnnulus(zip(
    x_centers_472_F356W, y_centers_472_F356W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_472_F356W = ApertureStats(data_F356W, random_annulus_472_F356W)
annulus_bkg_mean_472_F356W = apertures_stats_472_F356W.mean

total_random_bkg_472_F356W = annulus_bkg_mean_472_F356W * \
    random_apertures_472_F356W_area

phot_table_472_F356W = aperture_photometry(
    data_F356W, random_apertures_472_F356W)
phot_bkgsub_472_F356W = phot_table_472_F356W['aperture_sum'] - \
    total_random_bkg_472_F356W
phot_table_472_F356W['total_bkg'] = total_random_bkg_472_F356W
phot_table_472_F356W['aperture_sum_bkgsub'] = phot_bkgsub_472_F356W

for col in phot_table_472_F356W.colnames:
    # for consistent table output
    phot_table_472_F356W[col].info.format = '%.8g'

print(phot_table_472_F356W)

fluxes_472_F356W = phot_table_472_F356W['aperture_sum_bkgsub']
print(len(fluxes_472_F356W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_472_F356W)))
fluxes_472_F356W = fluxes_472_F356W[np.where(
    fluxes_472_F356W < 1*np.std(fluxes_472_F356W))]
print(len(fluxes_472_F356W))
fluxes_472_F356W_std = np.std(fluxes_472_F356W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_472_F356W_std))

# CEERSYJ-9586559217

cutout_217_F356W = data_F356W[int(np.floor(x_min_217_F356W)):int(np.ceil(
    x_max_217_F356W)), int(np.floor(y_min_217_F356W)):int(np.ceil(y_max_217_F356W))]

lower_x_217_F356W, upper_x_217_F356W = int(np.ceil(
    x_min_217_F356W + aperture_radius*3)), int(np.floor(x_max_217_F356W - aperture_radius*3))
lower_y_217_F356W, upper_y_217_F356W = int(np.ceil(
    y_min_217_F356W + aperture_radius*3)), int(np.floor(y_max_217_F356W - aperture_radius*3))

x_centers_217_F356W = [random.randrange(
    start=lower_x_217_F356W, stop=upper_x_217_F356W) for i in range(N_apertures)]
y_centers_217_F356W = [random.randrange(
    start=lower_y_217_F356W, stop=upper_y_217_F356W) for i in range(N_apertures)]

random_apertures_217_F356W = CircularAperture(
    zip(x_centers_217_F356W, y_centers_217_F356W), r=aperture_radius)
random_apertures_217_F356W_area = aperture.area

random_annulus_217_F356W = CircularAnnulus(zip(
    x_centers_217_F356W, y_centers_217_F356W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_217_F356W = ApertureStats(data_F356W, random_annulus_217_F356W)
annulus_bkg_mean_217_F356W = apertures_stats_217_F356W.mean

total_random_bkg_217_F356W = annulus_bkg_mean_217_F356W * \
    random_apertures_217_F356W_area

phot_table_217_F356W = aperture_photometry(
    data_F356W, random_apertures_217_F356W)
phot_bkgsub_217_F356W = phot_table_217_F356W['aperture_sum'] - \
    total_random_bkg_217_F356W
phot_table_217_F356W['total_bkg'] = total_random_bkg_217_F356W
phot_table_217_F356W['aperture_sum_bkgsub'] = phot_bkgsub_217_F356W

for col in phot_table_217_F356W.colnames:
    # for consistent table output
    phot_table_217_F356W[col].info.format = '%.8g'

print(phot_table_217_F356W)

fluxes_217_F356W = phot_table_217_F356W['aperture_sum_bkgsub']
print(len(fluxes_217_F356W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_217_F356W)))
fluxes_217_F356W = fluxes_217_F356W[np.where(
    fluxes_217_F356W < 1*np.std(fluxes_217_F356W))]
print(len(fluxes_217_F356W))
fluxes_217_F356W_std = np.std(fluxes_217_F356W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_217_F356W_std))

# CEERSYJ-0012959481

cutout_481_F356W = data_F356W[int(np.floor(x_min_481_F356W)):int(np.ceil(
    x_max_481_F356W)), int(np.floor(y_min_481_F356W)):int(np.ceil(y_max_481_F356W))]

lower_x_481_F356W, upper_x_481_F356W = int(np.ceil(
    x_min_481_F356W + aperture_radius*3)), int(np.floor(x_max_481_F356W - aperture_radius*3))
lower_y_481_F356W, upper_y_481_F356W = int(np.ceil(
    y_min_481_F356W + aperture_radius*3)), int(np.floor(y_max_481_F356W - aperture_radius*3))

x_centers_481_F356W = [random.randrange(
    start=lower_x_481_F356W, stop=upper_x_481_F356W) for i in range(N_apertures)]
y_centers_481_F356W = [random.randrange(
    start=lower_y_481_F356W, stop=upper_y_481_F356W) for i in range(N_apertures)]

random_apertures_481_F356W = CircularAperture(
    zip(x_centers_481_F356W, y_centers_481_F356W), r=aperture_radius)
random_apertures_481_F356W_area = aperture.area

random_annulus_481_F356W = CircularAnnulus(zip(
    x_centers_481_F356W, y_centers_481_F356W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_481_F356W = ApertureStats(data_F356W, random_annulus_481_F356W)
annulus_bkg_mean_481_F356W = apertures_stats_481_F356W.mean

total_random_bkg_481_F356W = annulus_bkg_mean_481_F356W * \
    random_apertures_481_F356W_area

phot_table_481_F356W = aperture_photometry(
    data_F356W, random_apertures_481_F356W)
phot_bkgsub_481_F356W = phot_table_481_F356W['aperture_sum'] - \
    total_random_bkg_481_F356W
phot_table_481_F356W['total_bkg'] = total_random_bkg_481_F356W
phot_table_481_F356W['aperture_sum_bkgsub'] = phot_bkgsub_481_F356W

for col in phot_table_481_F356W.colnames:
    # for consistent table output
    phot_table_481_F356W[col].info.format = '%.8g'

print(phot_table_481_F356W)

fluxes_481_F356W = phot_table_481_F356W['aperture_sum_bkgsub']
print(len(fluxes_481_F356W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_481_F356W)))
fluxes_481_F356W = fluxes_481_F356W[np.where(
    fluxes_481_F356W < 1*np.std(fluxes_481_F356W))]
print(len(fluxes_481_F356W))
fluxes_481_F356W_std = np.std(fluxes_481_F356W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_481_F356W_std))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Gráfico de apertura de puntos azarosos cercanos a
CEERSYJ-0012159472, CEERSYJ-9586559217 y CEERSYJ-0012959481
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F356W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F356W, interval=ZScaleInterval()))
central_aperture_F356W_472.plot(color='yellow', lw=2)
random_apertures_472_F356W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlim((int(np.floor(x_min_472_F356W)), int(np.ceil(x_max_472_F356W))))
plt.ylim((int(np.floor(y_min_472_F356W)), int(np.ceil(y_max_472_F356W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F356W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F356W, interval=ZScaleInterval()))
central_aperture_F356W_217.plot(color='yellow', lw=2)
random_apertures_217_F356W.plot(color='lime', lw=2)
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlim((int(np.floor(x_min_217_F356W)), int(np.ceil(x_max_217_F356W))))
plt.ylim((int(np.floor(y_min_217_F356W)), int(np.ceil(y_max_217_F356W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F356W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F356W, interval=ZScaleInterval()))
central_aperture_F356W_481.plot(color='yellow', lw=2)
random_apertures_481_F356W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlim((int(np.floor(x_min_481_F356W)), int(np.ceil(x_max_481_F356W))))
plt.ylim((int(np.floor(y_min_481_F356W)), int(np.ceil(y_max_481_F356W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F356W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F356W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
for i, ax in enumerate(axs):
    axs[i].legend(
        ['Apertura del objeto', 'Apertura de puntos azarosos'], loc='upper right')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Lectura archivos .fits de filtro F410W
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

hdul_F410W = fits.open(
    '/home/javier/Escritorio/TTB/nircam1/hlsp_ceers_jwst_nircam_nircam1_f410m_dr0.5_i2d.fits')
hdr_F410W = hdul_F410W[0].header
data_F410W = hdul_F410W[1].data

pixel_scale = 0.03  # arcsecs/pixel


"""""""""""""""""""""""""""""""""""""""""""""
Generación de apertura y anillos de fondo
"""""""""""""""""""""""""""""""""""""""""""""

# CEERSYJ-0012159472

aperture_diameter_arcsecs_472 = 0.3  # 0.35
aperture_radius_472 = aperture_diameter_arcsecs_472 / pixel_scale / 2

xc_472_F410W, yc_472_F410W = 2563.17, 2677.76
position_472_F410W = (xc_472_F410W, yc_472_F410W)

central_aperture_F410W_472 = CircularAperture(
    position_472_F410W, r=aperture_radius_472)
annulus_aperture_F410W_472 = CircularAnnulus(
    position_472_F410W, r_in=aperture_radius_472*2, r_out=aperture_radius_472*3)

delta_arcsecs_472_F410W = 3
delta_pix_472_F410W = delta_arcsecs_472_F410W / pixel_scale
x_min_472_F410W = xc_472_F410W - delta_pix_472_F410W
x_max_472_F410W = xc_472_F410W + delta_pix_472_F410W
y_min_472_F410W = yc_472_F410W - delta_pix_472_F410W
y_max_472_F410W = yc_472_F410W + delta_pix_472_F410W

# CEERSYJ-9586559217

aperture_diameter_arcsecs_217 = 0.3  # 0.208
aperture_radius_217 = aperture_diameter_arcsecs_217 / pixel_scale / 2

xc_217_F410W, yc_217_F410W = 3708.89, 2710.52
position_217_F410W = (xc_217_F410W, yc_217_F410W)

central_aperture_F410W_217 = CircularAperture(
    position_217_F410W, r=aperture_radius_217)
annulus_aperture_F410W_217 = CircularAnnulus(
    position_217_F410W, r_in=aperture_radius_217*2, r_out=aperture_radius_217*3)

delta_arcsecs_217_F410W = 3
delta_pix_217_F410W = delta_arcsecs_217_F410W / pixel_scale
x_min_217_F410W = xc_217_F410W - delta_pix_217_F410W
x_max_217_F410W = xc_217_F410W + delta_pix_217_F410W
y_min_217_F410W = yc_217_F410W - delta_pix_217_F410W
y_max_217_F410W = yc_217_F410W + delta_pix_217_F410W

# CEERSYJ-0012959481

aperture_diameter_arcsecs_481 = 0.3  # 0.336
aperture_radius_481 = aperture_diameter_arcsecs_481 / pixel_scale / 2

xc_481_F410W, yc_481_F410W = 2526.86, 2677.60
position_481_F410W = (xc_481_F410W, yc_481_F410W)

central_aperture_F410W_481 = CircularAperture(
    position_481_F410W, r=aperture_radius_481)
annulus_aperture_F410W_481 = CircularAnnulus(
    position_481_F410W, r_in=aperture_radius_481*2, r_out=aperture_radius_481*3)

delta_arcsecs_481_F410W = 3
delta_pix_481_F410W = delta_arcsecs_481_F410W / pixel_scale
x_min_481_F410W = xc_481_F410W - delta_pix_481_F410W
x_max_481_F410W = xc_481_F410W + delta_pix_481_F410W
y_min_481_F410W = yc_481_F410W - delta_pix_481_F410W
y_max_481_F410W = yc_481_F410W + delta_pix_481_F410W


"""""""""""""""""""""""""""""""""""""""
Gráfico de aperturas a las 3 galaxias
"""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F410W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F410W, interval=ZScaleInterval()))
central_aperture_F410W_472.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F410W_472.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_472_F410W, x_max_472_F410W, y_min_472_F410W, y_max_472_F410W])

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F410W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F410W, interval=ZScaleInterval()))
central_aperture_F410W_217.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F410W_217.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_217_F410W, x_max_217_F410W, y_min_217_F410W, y_max_217_F410W])

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F410W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F410W, interval=ZScaleInterval()))
central_aperture_F410W_481.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F410W_481.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_481_F410W, x_max_481_F410W, y_min_481_F410W, y_max_481_F410W])

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F410W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F410W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Fotometria de apertura con sustracción de fondo a CEERSYJ-0012159472,
CEERSYJ-9586559217 y CEERSYJ-0012959481 con una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

positions = [position_472_F410W, position_217_F410W, position_481_F410W]

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2

aperture = CircularAperture(positions, r=aperture_radius)
aperture_area = aperture.area

annulus_aperture = CircularAnnulus(
    positions, r_in=aperture_radius*2, r_out=aperture_radius*3)

aperstats = ApertureStats(data_F410W, annulus_aperture)
bkg_mean = aperstats.mean

total_bkg = bkg_mean * aperture_area

phot_table_F410W = aperture_photometry(data_F410W, aperture)
phot_bkgsub = phot_table_F410W['aperture_sum'] - total_bkg
phot_table_F410W['total_bkg'] = total_bkg
phot_table_F410W['aperture_sum_bkgsub'] = phot_bkgsub

for col in phot_table_F410W.colnames:
    phot_table_F410W[col].info.format = '%.8g'  # for consistent table output

print(phot_table_F410W)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Obtención de errores de la fotometría de apertura con
sustracción de fondo y para una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2
N_apertures = 50

# CEERSYJ-0012159472

cutout_472_F410W = data_F410W[int(np.floor(x_min_472_F410W)):int(np.ceil(
    x_max_472_F410W)), int(np.floor(y_min_472_F410W)):int(np.ceil(y_max_472_F410W))]

lower_x_472_F410W, upper_x_472_F410W = int(np.ceil(
    x_min_472_F410W + aperture_radius*3)), int(np.floor(x_max_472_F410W - aperture_radius*3))
lower_y_472_F410W, upper_y_472_F410W = int(np.ceil(
    y_min_472_F410W + aperture_radius*3)), int(np.floor(y_max_472_F410W - aperture_radius*3))

x_centers_472_F410W = [random.randrange(
    start=lower_x_472_F410W, stop=upper_x_472_F410W) for i in range(N_apertures)]
y_centers_472_F410W = [random.randrange(
    start=lower_y_472_F410W, stop=upper_y_472_F410W) for i in range(N_apertures)]

random_apertures_472_F410W = CircularAperture(
    zip(x_centers_472_F410W, y_centers_472_F410W), r=aperture_radius)
random_apertures_472_F410W_area = aperture.area

random_annulus_472_F410W = CircularAnnulus(zip(
    x_centers_472_F410W, y_centers_472_F410W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_472_F410W = ApertureStats(data_F410W, random_annulus_472_F410W)
annulus_bkg_mean_472_F410W = apertures_stats_472_F410W.mean

total_random_bkg_472_F410W = annulus_bkg_mean_472_F410W * \
    random_apertures_472_F410W_area

phot_table_472_F410W = aperture_photometry(
    data_F410W, random_apertures_472_F410W)
phot_bkgsub_472_F410W = phot_table_472_F410W['aperture_sum'] - \
    total_random_bkg_472_F410W
phot_table_472_F410W['total_bkg'] = total_random_bkg_472_F410W
phot_table_472_F410W['aperture_sum_bkgsub'] = phot_bkgsub_472_F410W

for col in phot_table_472_F410W.colnames:
    # for consistent table output
    phot_table_472_F410W[col].info.format = '%.8g'

print(phot_table_472_F410W)

fluxes_472_F410W = phot_table_472_F410W['aperture_sum_bkgsub']
print(len(fluxes_472_F410W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_472_F410W)))
fluxes_472_F410W = fluxes_472_F410W[np.where(
    fluxes_472_F410W < 1*np.std(fluxes_472_F410W))]
print(len(fluxes_472_F410W))
fluxes_472_F410W_std = np.std(fluxes_472_F410W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_472_F410W_std))

# CEERSYJ-9586559217

cutout_217_F410W = data_F410W[int(np.floor(x_min_217_F410W)):int(np.ceil(
    x_max_217_F410W)), int(np.floor(y_min_217_F410W)):int(np.ceil(y_max_217_F410W))]

lower_x_217_F410W, upper_x_217_F410W = int(np.ceil(
    x_min_217_F410W + aperture_radius*3)), int(np.floor(x_max_217_F410W - aperture_radius*3))
lower_y_217_F410W, upper_y_217_F410W = int(np.ceil(
    y_min_217_F410W + aperture_radius*3)), int(np.floor(y_max_217_F410W - aperture_radius*3))

x_centers_217_F410W = [random.randrange(
    start=lower_x_217_F410W, stop=upper_x_217_F410W) for i in range(N_apertures)]
y_centers_217_F410W = [random.randrange(
    start=lower_y_217_F410W, stop=upper_y_217_F410W) for i in range(N_apertures)]

random_apertures_217_F410W = CircularAperture(
    zip(x_centers_217_F410W, y_centers_217_F410W), r=aperture_radius)
random_apertures_217_F410W_area = aperture.area

random_annulus_217_F410W = CircularAnnulus(zip(
    x_centers_217_F410W, y_centers_217_F410W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_217_F410W = ApertureStats(data_F410W, random_annulus_217_F410W)
annulus_bkg_mean_217_F410W = apertures_stats_217_F410W.mean

total_random_bkg_217_F410W = annulus_bkg_mean_217_F410W * \
    random_apertures_217_F410W_area

phot_table_217_F410W = aperture_photometry(
    data_F410W, random_apertures_217_F410W)
phot_bkgsub_217_F410W = phot_table_217_F410W['aperture_sum'] - \
    total_random_bkg_217_F410W
phot_table_217_F410W['total_bkg'] = total_random_bkg_217_F410W
phot_table_217_F410W['aperture_sum_bkgsub'] = phot_bkgsub_217_F410W

for col in phot_table_217_F410W.colnames:
    # for consistent table output
    phot_table_217_F410W[col].info.format = '%.8g'

print(phot_table_217_F410W)

fluxes_217_F410W = phot_table_217_F410W['aperture_sum_bkgsub']
print(len(fluxes_217_F410W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_217_F410W)))
fluxes_217_F410W = fluxes_217_F410W[np.where(
    fluxes_217_F410W < 1*np.std(fluxes_217_F410W))]
print(len(fluxes_217_F410W))
fluxes_217_F410W_std = np.std(fluxes_217_F410W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_217_F410W_std))

# CEERSYJ-0012959481

cutout_481_F410W = data_F410W[int(np.floor(x_min_481_F410W)):int(np.ceil(
    x_max_481_F410W)), int(np.floor(y_min_481_F410W)):int(np.ceil(y_max_481_F410W))]

lower_x_481_F410W, upper_x_481_F410W = int(np.ceil(
    x_min_481_F410W + aperture_radius*3)), int(np.floor(x_max_481_F410W - aperture_radius*3))
lower_y_481_F410W, upper_y_481_F410W = int(np.ceil(
    y_min_481_F410W + aperture_radius*3)), int(np.floor(y_max_481_F410W - aperture_radius*3))

x_centers_481_F410W = [random.randrange(
    start=lower_x_481_F410W, stop=upper_x_481_F410W) for i in range(N_apertures)]
y_centers_481_F410W = [random.randrange(
    start=lower_y_481_F410W, stop=upper_y_481_F410W) for i in range(N_apertures)]

random_apertures_481_F410W = CircularAperture(
    zip(x_centers_481_F410W, y_centers_481_F410W), r=aperture_radius)
random_apertures_481_F410W_area = aperture.area

random_annulus_481_F410W = CircularAnnulus(zip(
    x_centers_481_F410W, y_centers_481_F410W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_481_F410W = ApertureStats(data_F410W, random_annulus_481_F410W)
annulus_bkg_mean_481_F410W = apertures_stats_481_F410W.mean

total_random_bkg_481_F410W = annulus_bkg_mean_481_F410W * \
    random_apertures_481_F410W_area

phot_table_481_F410W = aperture_photometry(
    data_F410W, random_apertures_481_F410W)
phot_bkgsub_481_F410W = phot_table_481_F410W['aperture_sum'] - \
    total_random_bkg_481_F410W
phot_table_481_F410W['total_bkg'] = total_random_bkg_481_F410W
phot_table_481_F410W['aperture_sum_bkgsub'] = phot_bkgsub_481_F410W

for col in phot_table_481_F410W.colnames:
    # for consistent table output
    phot_table_481_F410W[col].info.format = '%.8g'

print(phot_table_481_F410W)

fluxes_481_F410W = phot_table_481_F410W['aperture_sum_bkgsub']
print(len(fluxes_481_F410W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_481_F410W)))
fluxes_481_F410W = fluxes_481_F410W[np.where(
    fluxes_481_F410W < 1*np.std(fluxes_481_F410W))]
print(len(fluxes_481_F410W))
fluxes_481_F410W_std = np.std(fluxes_481_F410W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_481_F410W_std))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Gráfico de apertura de puntos azarosos cercanos a
CEERSYJ-0012159472, CEERSYJ-9586559217 y CEERSYJ-0012959481
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F410W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F410W, interval=ZScaleInterval()))
central_aperture_F410W_472.plot(color='yellow', lw=2)
random_apertures_472_F410W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlim((int(np.floor(x_min_472_F410W)), int(np.ceil(x_max_472_F410W))))
plt.ylim((int(np.floor(y_min_472_F410W)), int(np.ceil(y_max_472_F410W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F410W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F410W, interval=ZScaleInterval()))
central_aperture_F410W_217.plot(color='yellow', lw=2)
random_apertures_217_F410W.plot(color='lime', lw=2)
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlim((int(np.floor(x_min_217_F410W)), int(np.ceil(x_max_217_F410W))))
plt.ylim((int(np.floor(y_min_217_F410W)), int(np.ceil(y_max_217_F410W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F410W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F410W, interval=ZScaleInterval()))
central_aperture_F410W_481.plot(color='yellow', lw=2)
random_apertures_481_F410W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlim((int(np.floor(x_min_481_F410W)), int(np.ceil(x_max_481_F410W))))
plt.ylim((int(np.floor(y_min_481_F410W)), int(np.ceil(y_max_481_F410W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F410W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F410W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
for i, ax in enumerate(axs):
    axs[i].legend(
        ['Apertura del objeto', 'Apertura de puntos azarosos'], loc='upper right')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Lectura archivos .fits de filtro F444W
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

hdul_F444W = fits.open(
    '/home/javier/Escritorio/TTB/nircam1/hlsp_ceers_jwst_nircam_nircam1_f444w_dr0.5_i2d.fits')
hdr_F444W = hdul_F444W[0].header
data_F444W = hdul_F444W[1].data

pixel_scale = 0.03  # arcsecs/pixel


"""""""""""""""""""""""""""""""""""""""""""""
Generación de apertura y anillos de fondo
"""""""""""""""""""""""""""""""""""""""""""""

# CEERSYJ-0012159472

aperture_diameter_arcsecs_472 = 0.3  # 0.35
aperture_radius_472 = aperture_diameter_arcsecs_472 / pixel_scale / 2

xc_472_F444W, yc_472_F444W = 2563.17, 2677.76
position_472_F444W = (xc_472_F444W, yc_472_F444W)

central_aperture_F444W_472 = CircularAperture(
    position_472_F444W, r=aperture_radius_472)
annulus_aperture_F444W_472 = CircularAnnulus(
    position_472_F444W, r_in=aperture_radius_472*2, r_out=aperture_radius_472*3)

delta_arcsecs_472_F444W = 3
delta_pix_472_F444W = delta_arcsecs_472_F444W / pixel_scale
x_min_472_F444W = xc_472_F444W - delta_pix_472_F444W
x_max_472_F444W = xc_472_F444W + delta_pix_472_F444W
y_min_472_F444W = yc_472_F444W - delta_pix_472_F444W
y_max_472_F444W = yc_472_F444W + delta_pix_472_F444W

# CEERSYJ-9586559217

aperture_diameter_arcsecs_217 = 0.3  # 0.208
aperture_radius_217 = aperture_diameter_arcsecs_217 / pixel_scale / 2

xc_217_F444W, yc_217_F444W = 3708.89, 2710.52
position_217_F444W = (xc_217_F444W, yc_217_F444W)

central_aperture_F444W_217 = CircularAperture(
    position_217_F444W, r=aperture_radius_217)
annulus_aperture_F444W_217 = CircularAnnulus(
    position_217_F444W, r_in=aperture_radius_217*2, r_out=aperture_radius_217*3)

delta_arcsecs_217_F444W = 3
delta_pix_217_F444W = delta_arcsecs_217_F444W / pixel_scale
x_min_217_F444W = xc_217_F444W - delta_pix_217_F444W
x_max_217_F444W = xc_217_F444W + delta_pix_217_F444W
y_min_217_F444W = yc_217_F444W - delta_pix_217_F444W
y_max_217_F444W = yc_217_F444W + delta_pix_217_F444W

# CEERSYJ-0012959481

aperture_diameter_arcsecs_481 = 0.3  # 0.336
aperture_radius_481 = aperture_diameter_arcsecs_481 / pixel_scale / 2

xc_481_F444W, yc_481_F444W = 2526.86, 2677.60
position_481_F444W = (xc_481_F444W, yc_481_F444W)

central_aperture_F444W_481 = CircularAperture(
    position_481_F444W, r=aperture_radius_481)
annulus_aperture_F444W_481 = CircularAnnulus(
    position_481_F444W, r_in=aperture_radius_481*2, r_out=aperture_radius_481*3)

delta_arcsecs_481_F444W = 3
delta_pix_481_F444W = delta_arcsecs_481_F444W / pixel_scale
x_min_481_F444W = xc_481_F444W - delta_pix_481_F444W
x_max_481_F444W = xc_481_F444W + delta_pix_481_F444W
y_min_481_F444W = yc_481_F444W - delta_pix_481_F444W
y_max_481_F444W = yc_481_F444W + delta_pix_481_F444W


"""""""""""""""""""""""""""""""""""""""
Gráfico de aperturas a las 3 galaxias
"""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F444W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F444W, interval=ZScaleInterval()))
central_aperture_F444W_472.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F444W_472.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_472_F444W, x_max_472_F444W, y_min_472_F444W, y_max_472_F444W])

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F444W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F444W, interval=ZScaleInterval()))
central_aperture_F444W_217.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F444W_217.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_217_F444W, x_max_217_F444W, y_min_217_F444W, y_max_217_F444W])

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F444W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F444W, interval=ZScaleInterval()))
central_aperture_F444W_481.plot(
    color='yellow', lw=2, label='Apertura del objeto')
annulus_aperture_F444W_481.plot(
    color='r', lw=2, label='Anillos de fondo de cielo\ncercano al objeto')
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()
plt.axis([x_min_481_F444W, x_max_481_F444W, y_min_481_F444W, y_max_481_F444W])

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F444W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F444W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Fotometria de apertura con sustracción de fondo a CEERSYJ-0012159472,
CEERSYJ-9586559217 y CEERSYJ-0012959481 con una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

positions = [position_472_F444W, position_217_F444W, position_481_F444W]

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2

aperture = CircularAperture(positions, r=aperture_radius)
aperture_area = aperture.area

annulus_aperture = CircularAnnulus(
    positions, r_in=aperture_radius*2, r_out=aperture_radius*3)

aperstats = ApertureStats(data_F444W, annulus_aperture)
bkg_mean = aperstats.mean

total_bkg = bkg_mean * aperture_area

phot_table_F444W = aperture_photometry(data_F444W, aperture)
phot_bkgsub = phot_table_F444W['aperture_sum'] - total_bkg
phot_table_F444W['total_bkg'] = total_bkg
phot_table_F444W['aperture_sum_bkgsub'] = phot_bkgsub

for col in phot_table_F444W.colnames:
    phot_table_F444W[col].info.format = '%.8g'  # for consistent table output

print(phot_table_F444W)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Obtención de errores de la fotometría de apertura con
sustracción de fondo y para una apertura de objeto definida
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

aperture_diameter_arcsecs = 0.3
pixel_scale = 0.03  # arcsecs/pixel
aperture_radius = aperture_diameter_arcsecs / pixel_scale / 2
N_apertures = 50

# CEERSYJ-0012159472

cutout_472_F444W = data_F444W[int(np.floor(x_min_472_F444W)):int(np.ceil(
    x_max_472_F444W)), int(np.floor(y_min_472_F444W)):int(np.ceil(y_max_472_F444W))]

lower_x_472_F444W, upper_x_472_F444W = int(np.ceil(
    x_min_472_F444W + aperture_radius*3)), int(np.floor(x_max_472_F444W - aperture_radius*3))
lower_y_472_F444W, upper_y_472_F444W = int(np.ceil(
    y_min_472_F444W + aperture_radius*3)), int(np.floor(y_max_472_F444W - aperture_radius*3))

x_centers_472_F444W = [random.randrange(
    start=lower_x_472_F444W, stop=upper_x_472_F444W) for i in range(N_apertures)]
y_centers_472_F444W = [random.randrange(
    start=lower_y_472_F444W, stop=upper_y_472_F444W) for i in range(N_apertures)]

random_apertures_472_F444W = CircularAperture(
    zip(x_centers_472_F444W, y_centers_472_F444W), r=aperture_radius)
random_apertures_472_F444W_area = aperture.area

random_annulus_472_F444W = CircularAnnulus(zip(
    x_centers_472_F444W, y_centers_472_F444W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_472_F444W = ApertureStats(data_F444W, random_annulus_472_F444W)
annulus_bkg_mean_472_F444W = apertures_stats_472_F444W.mean

total_random_bkg_472_F444W = annulus_bkg_mean_472_F444W * \
    random_apertures_472_F444W_area

phot_table_472_F444W = aperture_photometry(
    data_F444W, random_apertures_472_F444W)
phot_bkgsub_472_F444W = phot_table_472_F444W['aperture_sum'] - \
    total_random_bkg_472_F444W
phot_table_472_F444W['total_bkg'] = total_random_bkg_472_F444W
phot_table_472_F444W['aperture_sum_bkgsub'] = phot_bkgsub_472_F444W

for col in phot_table_472_F444W.colnames:
    # for consistent table output
    phot_table_472_F444W[col].info.format = '%.8g'

print(phot_table_472_F444W)

fluxes_472_F444W = phot_table_472_F444W['aperture_sum_bkgsub']
print(len(fluxes_472_F444W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_472_F444W)))
fluxes_472_F444W = fluxes_472_F444W[np.where(
    fluxes_472_F444W < 1*np.std(fluxes_472_F444W))]
print(len(fluxes_472_F444W))
fluxes_472_F444W_std = np.std(fluxes_472_F444W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_472_F444W_std))

# CEERSYJ-9586559217

cutout_217_F444W = data_F444W[int(np.floor(x_min_217_F444W)):int(np.ceil(
    x_max_217_F444W)), int(np.floor(y_min_217_F444W)):int(np.ceil(y_max_217_F444W))]

lower_x_217_F444W, upper_x_217_F444W = int(np.ceil(
    x_min_217_F444W + aperture_radius*3)), int(np.floor(x_max_217_F444W - aperture_radius*3))
lower_y_217_F444W, upper_y_217_F444W = int(np.ceil(
    y_min_217_F444W + aperture_radius*3)), int(np.floor(y_max_217_F444W - aperture_radius*3))

x_centers_217_F444W = [random.randrange(
    start=lower_x_217_F444W, stop=upper_x_217_F444W) for i in range(N_apertures)]
y_centers_217_F444W = [random.randrange(
    start=lower_y_217_F444W, stop=upper_y_217_F444W) for i in range(N_apertures)]

random_apertures_217_F444W = CircularAperture(
    zip(x_centers_217_F444W, y_centers_217_F444W), r=aperture_radius)
random_apertures_217_F444W_area = aperture.area

random_annulus_217_F444W = CircularAnnulus(zip(
    x_centers_217_F444W, y_centers_217_F444W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_217_F444W = ApertureStats(data_F444W, random_annulus_217_F444W)
annulus_bkg_mean_217_F444W = apertures_stats_217_F444W.mean

total_random_bkg_217_F444W = annulus_bkg_mean_217_F444W * \
    random_apertures_217_F444W_area

phot_table_217_F444W = aperture_photometry(
    data_F444W, random_apertures_217_F444W)
phot_bkgsub_217_F444W = phot_table_217_F444W['aperture_sum'] - \
    total_random_bkg_217_F444W
phot_table_217_F444W['total_bkg'] = total_random_bkg_217_F444W
phot_table_217_F444W['aperture_sum_bkgsub'] = phot_bkgsub_217_F444W

for col in phot_table_217_F444W.colnames:
    # for consistent table output
    phot_table_217_F444W[col].info.format = '%.8g'

print(phot_table_217_F444W)

fluxes_217_F444W = phot_table_217_F444W['aperture_sum_bkgsub']
print(len(fluxes_217_F444W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_217_F444W)))
fluxes_217_F444W = fluxes_217_F444W[np.where(
    fluxes_217_F444W < 1*np.std(fluxes_217_F444W))]
print(len(fluxes_217_F444W))
fluxes_217_F444W_std = np.std(fluxes_217_F444W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_217_F444W_std))

# CEERSYJ-0012959481

cutout_481_F444W = data_F444W[int(np.floor(x_min_481_F444W)):int(np.ceil(
    x_max_481_F444W)), int(np.floor(y_min_481_F444W)):int(np.ceil(y_max_481_F444W))]

lower_x_481_F444W, upper_x_481_F444W = int(np.ceil(
    x_min_481_F444W + aperture_radius*3)), int(np.floor(x_max_481_F444W - aperture_radius*3))
lower_y_481_F444W, upper_y_481_F444W = int(np.ceil(
    y_min_481_F444W + aperture_radius*3)), int(np.floor(y_max_481_F444W - aperture_radius*3))

x_centers_481_F444W = [random.randrange(
    start=lower_x_481_F444W, stop=upper_x_481_F444W) for i in range(N_apertures)]
y_centers_481_F444W = [random.randrange(
    start=lower_y_481_F444W, stop=upper_y_481_F444W) for i in range(N_apertures)]

random_apertures_481_F444W = CircularAperture(
    zip(x_centers_481_F444W, y_centers_481_F444W), r=aperture_radius)
random_apertures_481_F444W_area = aperture.area

random_annulus_481_F444W = CircularAnnulus(zip(
    x_centers_481_F444W, y_centers_481_F444W), r_in=aperture_radius*2, r_out=aperture_radius*3)

apertures_stats_481_F444W = ApertureStats(data_F444W, random_annulus_481_F444W)
annulus_bkg_mean_481_F444W = apertures_stats_481_F444W.mean

total_random_bkg_481_F444W = annulus_bkg_mean_481_F444W * \
    random_apertures_481_F444W_area

phot_table_481_F444W = aperture_photometry(
    data_F444W, random_apertures_481_F444W)
phot_bkgsub_481_F444W = phot_table_481_F444W['aperture_sum'] - \
    total_random_bkg_481_F444W
phot_table_481_F444W['total_bkg'] = total_random_bkg_481_F444W
phot_table_481_F444W['aperture_sum_bkgsub'] = phot_bkgsub_481_F444W

for col in phot_table_481_F444W.colnames:
    # for consistent table output
    phot_table_481_F444W[col].info.format = '%.8g'

print(phot_table_481_F444W)

fluxes_481_F444W = phot_table_481_F444W['aperture_sum_bkgsub']
print(len(fluxes_481_F444W))
print('The std of the fluxes before removing points where the flux is not corresponding to background: {:.3f}'.format(
    np.std(fluxes_481_F444W)))
fluxes_481_F444W = fluxes_481_F444W[np.where(
    fluxes_481_F444W < 1*np.std(fluxes_481_F444W))]
print(len(fluxes_481_F444W))
fluxes_481_F444W_std = np.std(fluxes_481_F444W)
print('The std of the fluxes after removing points where the flux is not corresponding to background: {:.3f}'.format(
    fluxes_481_F444W_std))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Gráfico de apertura de puntos azarosos cercanos a
CEERSYJ-0012159472, CEERSYJ-9586559217 y CEERSYJ-0012959481
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

fig = plt.figure(constrained_layout=True, figsize=(24, 8))

# CEERSYJ-0012159472

ax1 = plt.subplot(1, 3, 1)

plt.imshow(data_F444W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F444W, interval=ZScaleInterval()))
central_aperture_F444W_472.plot(color='yellow', lw=2)
random_apertures_472_F444W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012159472', fontsize=16)
plt.xlim((int(np.floor(x_min_472_F444W)), int(np.ceil(x_max_472_F444W))))
plt.ylim((int(np.floor(y_min_472_F444W)), int(np.ceil(y_max_472_F444W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-9586559217

ax2 = plt.subplot(1, 3, 2)

plt.imshow(data_F444W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F444W, interval=ZScaleInterval()))
central_aperture_F444W_217.plot(color='yellow', lw=2)
random_apertures_217_F444W.plot(color='lime', lw=2)
plt.title('CEERSYJ-9586559217', fontsize=16)
plt.xlim((int(np.floor(x_min_217_F444W)), int(np.ceil(x_max_217_F444W))))
plt.ylim((int(np.floor(y_min_217_F444W)), int(np.ceil(y_max_217_F444W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

# CEERSYJ-0012959481

ax3 = plt.subplot(1, 3, 3)

plt.imshow(data_F444W, cmap='gray', origin='lower',
           norm=ImageNormalize(data_F444W, interval=ZScaleInterval()))
central_aperture_F444W_481.plot(color='yellow', lw=2)
random_apertures_481_F444W.plot(color='lime', lw=2)
plt.title('CEERSYJ-0012959481', fontsize=16)
plt.xlim((int(np.floor(x_min_481_F444W)), int(np.ceil(x_max_481_F444W))))
plt.ylim((int(np.floor(y_min_481_F444W)), int(np.ceil(y_max_481_F444W))))
plt.xlabel(r'Número de columna [pixel]')
plt.ylabel(r'Número de fila [pixel]')
plt.grid()
plt.legend()

axs = [ax1, ax2, ax3]

sm = plt.cm.ScalarMappable(cmap='gray', norm=ImageNormalize(
    data_F444W, interval=ZScaleInterval()))

fig.suptitle('Observación a través del instrumento NIRCam1 con filtro\nfotométrico F444W e imagen escalada a ZScale de:', fontsize=20, y=0.87)
clb = fig.colorbar(sm, ax=axs, fraction=0.0151, pad=0.04)
clb.ax.set_title("Brillo superficial\n" + r'$\left[ \frac{MJy}{sr} \right]$')
for i, ax in enumerate(axs):
    axs[i].legend(
        ['Apertura del objeto', 'Apertura de puntos azarosos'], loc='upper right')
plt.tight_layout
plt.show()
plt.clf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Gráfico de la distribución espectral de energía (SEP)
CEERSYJ-0012159472, CEERSYJ-9586559217 y CEERSYJ-0012959481
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Longitud de onda

wavelength_ref_ang = np.array([5921.88, 8045.53, 11542.61, 12486.07, 13923.21,
                              15007.44, 15370.34, 19886.48, 27617.40, 35683.62, 40822.38, 44043.14])
wavelength_ref_um = wavelength_ref_ang / 10000
print(wavelength_ref_um)

# Densidad de flujo

density_flux_3_galaxies = [phot_table_F606W['aperture_sum_bkgsub'], phot_table_F814W['aperture_sum_bkgsub'], phot_table_F115W['aperture_sum_bkgsub'], phot_table_F125W['aperture_sum_bkgsub'], phot_table_F140W['aperture_sum_bkgsub'], phot_table_F150W['aperture_sum_bkgsub'],
                           phot_table_F160W['aperture_sum_bkgsub'], phot_table_F200W['aperture_sum_bkgsub'], phot_table_F277W['aperture_sum_bkgsub'], phot_table_F356W['aperture_sum_bkgsub'], phot_table_F410W['aperture_sum_bkgsub'], phot_table_F444W['aperture_sum_bkgsub']]
density_flux_472 = []
density_flux_217 = []
density_flux_481 = []
for values in density_flux_3_galaxies:
    density_flux_472.append(values[0])
    density_flux_217.append(values[1])
    density_flux_481.append(values[2])

print(density_flux_472)
print(density_flux_217)
print(density_flux_481)

# Error de la densidad de flujo

density_flux_error_472 = [fluxes_472_F606W_std, fluxes_472_F814W_std, fluxes_472_F115W_std, fluxes_472_F125W_std, fluxes_472_F140W_std,
                          fluxes_472_F150W_std, fluxes_472_F160W_std, fluxes_472_F200W_std, fluxes_472_F277W_std, fluxes_472_F356W_std, fluxes_472_F410W_std, fluxes_472_F444W_std]
density_flux_error_217 = [fluxes_217_F606W_std, fluxes_217_F814W_std, fluxes_217_F115W_std, fluxes_217_F125W_std, fluxes_217_F140W_std,
                          fluxes_217_F150W_std, fluxes_217_F160W_std, fluxes_217_F200W_std, fluxes_217_F277W_std, fluxes_217_F356W_std, fluxes_217_F410W_std, fluxes_217_F444W_std]
density_flux_error_481 = [fluxes_481_F606W_std, fluxes_481_F814W_std, fluxes_481_F115W_std, fluxes_481_F125W_std, fluxes_481_F140W_std,
                          fluxes_481_F150W_std, fluxes_481_F160W_std, fluxes_481_F200W_std, fluxes_481_F277W_std, fluxes_481_F356W_std, fluxes_481_F410W_std, fluxes_481_F444W_std]

print(density_flux_error_472)
print(density_flux_error_217)
print(density_flux_error_481)

fig, ax = plt.subplots()

trans_472 = Affine2D().translate(-0.0175, 0.0) + ax.transData
trans_217 = Affine2D().translate(-0.0, 0.0) + ax.transData
trans_481 = Affine2D().translate(+0.0175, 0.0) + ax.transData
er1 = ax.errorbar(wavelength_ref_um, density_flux_472, yerr=density_flux_error_472, fmt='b.', ecolor='b',
                  capsize=5, capthick=1, label='CEERSYJ-0012159472 (' + r'$z_{phot}$' + ' ~ 8.9)', transform=trans_472)
er2 = ax.errorbar(wavelength_ref_um, density_flux_217, yerr=density_flux_error_217, fmt='g.', ecolor='g',
                  capsize=5, capthick=1, label='CEERSYJ-9586559217 (' + r'$z_{phot}$' + ' ~ 9.2)', transform=trans_217)
er3 = ax.errorbar(wavelength_ref_um, density_flux_481, yerr=density_flux_error_481, fmt='r.', ecolor='r',
                  capsize=5, capthick=1, label='CEERSYJ-0012959481 (' + r'$z_{phot}$' + ' ~ 9.9)', transform=trans_481)

plt.xlabel(r'$\lambda_{pivot}$ [$\mu m$]')
plt.xlim([0.5, 4.5])
plt.ylabel("Densidad de flujo" + r'$\left[ \frac{MJy}{sr} \right]$')
plt.legend()
plt.title('Distribución espectral de energía', fontsize=16)
plt.grid()
plt.show()
plt.clf()