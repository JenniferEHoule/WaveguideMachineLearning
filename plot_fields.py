""" plot_fields.py

Plot the waveguide cross-sections

Author: Jennifer Houle
Date: 4/22/2022

"""

from matplotlib import pyplot as plt
from matplotlib import gridspec

def plot_e_fields(data, mode, m, n, waveguide):

    if mode:
        label = 'TE'
    else:
        label = 'TM'

    plt.rcParams.update({'font.size': 18})
    plt.rcParams["figure.autolayout"] = True
    fig, ax1 = plt.subplots(figsize=(14, 12))

    gs = gridspec.GridSpec(3, 2)
    ax1 = plt.subplot(gs[0])
    cb1 = ax1.imshow((data[0, :, :]), extent=[0, waveguide.a, 0, waveguide.b])
    ax1.set_title(f"{label}$_{m}$$_{n}$ Mode, |E$_x$|")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    fig.colorbar(cb1)

    ax2 = plt.subplot(gs[1])
    cb2 = ax2.imshow((data[1, :, :]), extent=[0, waveguide.a, 0, waveguide.b])
    ax2.set_title(f"{label}$_{m}$$_{n}$ Mode, \u2220E$_x$ [rad]")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    fig.colorbar(cb2)

    ax3 = plt.subplot(gs[2])
    cb3 = ax3.imshow((data[2, :, :]), extent=[0, waveguide.a, 0, waveguide.b])
    ax3.set_title(f"{label}$_{m}$$_{n}$ Mode, |E$_y$|")
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")
    fig.colorbar(cb3)

    ax4 = plt.subplot(gs[3])
    cb4 = ax4.imshow((data[3, :, :]), extent=[0, waveguide.a, 0, waveguide.b])
    ax4.set_title(f"{label}$_{m}$$_{n}$ Mode, \u2220E$_y$ [rad]")
    ax4.set_xlabel("x [m]")
    ax4.set_ylabel("y [m]")
    fig.colorbar(cb4)

    ax5 = plt.subplot(gs[4])
    cb5 = ax5.imshow((data[4, :, :]), extent=[0, waveguide.a, 0, waveguide.b])
    ax5.set_title(f"{label}$_{m}$$_{n}$ Mode, |E$_z$|")
    ax5.set_xlabel("x [m]")
    ax5.set_ylabel("y [m]")
    fig.colorbar(cb5)

    ax6 = plt.subplot(gs[5])
    cb6 = ax6.imshow((data[5, :, :]), extent=[0, waveguide.a, 0, waveguide.b])
    ax6.set_title(f"{label}$_{m}$$_{n}$ Mode, \u2220E$_z$ [rad]")
    ax6.set_xlabel("x [m]")
    ax6.set_ylabel("y [m]")
    fig.colorbar(cb6)

    fig.tight_layout()
    plt.show()


def plot_ex_field(data, mode, m, n, waveguide):

    if mode:
        label = 'TE'
    else:
        label = 'TM'

    plt.rcParams.update({'font.size': 18})
    plt.rcParams["figure.autolayout"] = True
    fig, ax1 = plt.subplots(figsize=(8, 5))

    cb1 = ax1.imshow((data[0, :, :]), extent=[0, waveguide.a, 0, waveguide.b])
    ax1.set_title(f"{label}$_{m}$$_{n}$ Mode, |E$_x$|")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    fig.colorbar(cb1)

    fig.tight_layout()
    plt.show()


def plot_h_fields(data, mode, m, n, waveguide):

    if mode:
        label = 'TE'
    else:
        label = 'TM'

    plt.rcParams.update({'font.size': 18})
    plt.rcParams["figure.autolayout"] = True
    fig, ax1 = plt.subplots(figsize=(14, 12))
    # plt.suptitle(f'{freq:.2e} Hz')

    gs = gridspec.GridSpec(3, 2)
    ax1 = plt.subplot(gs[0])
    cb1 = ax1.imshow(data[6, :, :], extent=[0, waveguide.a, 0, waveguide.b])
    ax1.set_title(f"{label}$_{m}$$_{n}$ Mode, |H$_x$|")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    fig.colorbar(cb1)

    ax2 = plt.subplot(gs[1])
    cb2 = ax2.imshow(data[7, :, :], extent=[0, waveguide.a, 0, waveguide.b])
    ax2.set_title(f"{label}$_{m}$$_{n}$ Mode, \u2220H$_x$ [rad]")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    fig.colorbar(cb2)

    ax3 = plt.subplot(gs[2])
    cb3 = ax3.imshow(data[8, :, :], extent=[0, waveguide.a, 0, waveguide.b])
    ax3.set_title(f"{label}$_{m}$$_{n}$ Mode, |H$_y$|")
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")
    fig.colorbar(cb3)

    ax4 = plt.subplot(gs[3])
    cb4 = ax4.imshow(data[9, :, :], extent=[0, waveguide.a, 0, waveguide.b])
    ax4.set_title(f"{label}$_{m}$$_{n}$ Mode, \u2220H$_y$ [rad]")
    ax4.set_xlabel("x [m]")
    ax4.set_ylabel("y [m]")
    fig.colorbar(cb4)

    ax5 = plt.subplot(gs[4])
    cb5 = ax5.imshow(data[10, :, :], extent=[0, waveguide.a, 0, waveguide.b])
    ax5.set_title(f"{label}$_{m}$$_{n}$ Mode, |H$_z$|")
    ax5.set_xlabel("x [m]")
    ax5.set_ylabel("y [m]")
    fig.colorbar(cb5)

    ax6 = plt.subplot(gs[5])
    cb6 = ax6.imshow(data[11, :, :], extent=[0, waveguide.a, 0, waveguide.b])
    ax6.set_title(f"{label}$_{m}$$_{n}$ Mode, \u2220H$_z$ [rad]")
    ax6.set_xlabel("x [m]")
    ax6.set_ylabel("y [m]")
    fig.colorbar(cb6)

    plt.tight_layout()
    plt.show()
