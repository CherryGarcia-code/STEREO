@echo off
cd /d E:\python_analysis\git_repos\STEREO

set PYTHON=C:\Users\Ben\AppData\Local\Programs\Python\Python310\python.exe

echo === 1/15 SuppFig1_STEREO_Validation ===
%PYTHON% revision\SuppFig_STEREO_Validation.py
echo.

echo === 2/15 SuppFig2_AllBehaviors_Development ===
%PYTHON% revision\SuppFig_AllBehaviors_Development.py
echo.

echo === 3/15 SuppFig3_iSPN_Opto_Transitions ===
%PYTHON% revision\SuppFig_iSPN_Opto_Transitions.py
echo.

echo === 4/15 SuppFig4_dSPN_Opto_BoutDur_TransProb ===
%PYTHON% revision\SuppFig_dSPN_Opto_BoutDur_TransProb.py
echo.

echo === 5/15 SuppFig5_DREADDs_AllCohorts ===
%PYTHON% revision\SuppFig_DREADDs_AllCohorts.py
echo.

echo === 6/15 SuppFig6_Stacked_Development ===
%PYTHON% revision\SuppFig_Stacked_Development.py
echo.

echo === 7/15 SuppFig7_Entropy_Manipulations ===
%PYTHON% revision\SuppFig_Entropy_Manipulations.py
echo.

echo === 8/15 SuppFig8_iSPN_Transient ===
%PYTHON% revision\SuppFig_iSPN_Transient.py
echo.

echo === 9/15 SuppFig9_dSPN_vs_iSPN_BoutDur ===
%PYTHON% revision\SuppFig_dSPN_vs_iSPN_BoutDur.py
echo.

echo === 10/15 SuppFig10_Photom_BoutCorrelation ===
%PYTHON% revision\SuppFig_Photom_BoutCorrelation.py
echo.

echo === 11/15 SuppFig11_Context_Specificity ===
%PYTHON% revision\SuppFig_Context_Specificity.py
echo.

echo === 12/15 SuppFig12_Velocity_Licking ===
%PYTHON% revision\SuppFig_Velocity_Licking.py
echo.

echo === 13/15 SuppFig13_Grooming_Photometry ===
%PYTHON% revision\SuppFig_Grooming_Photometry.py
echo.

echo === 14/15 SuppFig14_TimeWarped_SplashTest ===
%PYTHON% revision\SuppFig_TimeWarped_SplashTest.py
echo.

echo === 15/15 Impact scripts ===
%PYTHON% revision\Impact_BoutDecomposition.py
%PYTHON% revision\Impact_CompositeSummary.py
%PYTHON% revision\Impact_DoseResponse.py
%PYTHON% revision\Impact_ForestPlot.py
%PYTHON% revision\Impact_IndividualTrajectories.py
%PYTHON% revision\Impact_TransitionFlow.py
echo.

echo === ALL DONE ===
pause
