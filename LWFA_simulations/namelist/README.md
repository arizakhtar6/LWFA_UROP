This folder contains all namelists used for investigating LWFA with clustered and unclustered methane, as well as a baseline test run with background plasma.

'lwfa_clustered.py' produces clustered LWFA results, change the diagnostics to analyse whatever is required!

'lwfa_unclustered.py' is very similar, except the methane is at the background density.

'lwfa_20um' keeps all clusters within a 20um transverse distance from the laser pulse. You may want to adapt this so that clusters are instantiated for a longer x distance as the laser propagates over 100s of microns during the simulation. A similar method to what's done in 'lwfa_clustered.py' can be used.
