---
title: 'ModSpectra: A Python Package for Synthetic Spectra of Neutral and Ionized Gas'
tags:
  - Python
  - astronomy
  - interstellar medium
  - emission lines
  - position-position-velocity
  - kinematics
authors:
  - name: Dhanesh Krishnarao
    orcid: 0000-0002-7955-7359
    affiliation: 1
affiliations:
 - name: Department of Astronomy, University of Wisconsin-Madison
   index: 1
date: 31 January 2019
bibliography: paper.bib
---

# Summary

Understanding emisison line observations often requires an underlying model than can be used to translate observed radial velocity to a modeled line-of-sight distance. Standard methods of estimating kinematic distances can work well in some parts of the Milky Way, but poorly in others, especially when non-circular rotation becomes prevalent [@Reid:2016]. 

The ``modspectra`` package provides an easy way to create synthetic 3D data cubes 
of HI and H-Alpha spectra in the Galaxy. This package uses ``spectral-cube.SpectralCube`` [@spectral-cube] 
 to handle the data cube and the primary class, ``EmissionCube``, is built around this. The current main 
 features of modspectra are to complement
the work of Krishnarao, Benjamin, Haffner (2019) on a model of a Tilted 
Elliptical Gas Disk around Galactic Center inspried by @Burton&Liszt:1982. The basic functionality of the package provides the ability to define a density field in any coordinate frame and tranlate that information inot a position-position-velocity data cube of neutral gas HI-21cm emission or ionized gas H-Alpha emission using ``astropy.coordinates`` [@astropy].

Extinction corrections can be made in 3D for the optical emission using @Marshall:2006 dustmaps implented via `dustmaps` [@dustmaps] and `extinction` [@extinction]. 

An example of the type of emission models that can be created using this package is below. In this example, a rotating polar ring structure in HI is tilted along two axes and a resulting first-moment map is plotted, showing the mean velocity of the gas.

![Example polar ring model created using ``modspectra``.](figure.pdf)

# References