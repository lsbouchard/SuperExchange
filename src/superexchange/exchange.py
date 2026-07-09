"""Core superexchange calculation routines.

This module packages the original implementation from ``Exchange.ipynb`` so it
can be imported, tested, and reused from scripts.  The method names intentionally
remain compatible with the notebook.

Full superexchange calculations require Wolfram Client for Python and a local
Wolfram kernel.  Geometry, parsing, and tensor utility methods are available
without Wolfram.
"""

from __future__ import annotations

import itertools as it
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from wolframclient.evaluation import WolframLanguageSession, parallel_evaluate
    from wolframclient.language import wl, wlexpr
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    WolframLanguageSession = None
    parallel_evaluate = None
    wl = None
    wlexpr = None


def _require_wolfram() -> None:
    """Raise a clear error when the optional Wolfram dependency is unavailable."""
    if WolframLanguageSession is None:
        raise RuntimeError(
            "Full superexchange calculations require the optional "
            "'wolframclient' package and access to a local Wolfram kernel. "
            "Install with: pip install 'superexchange[wolfram]'"
        )


class Exchange:
    def __init__(self, molecule_data_sheet_name, wolfram_path = ''):
        self.wolfram_path = wolfram_path
        self.data_sheet_name = molecule_data_sheet_name
        self.found_t = {}
        self.found_de = {}
        self.charge_transfer_energies = {}
        #Should be a panda containing the molecule data instructions.
        self.molecule = []
        self.get_Molecule_Data(molecule_data_sheet_name)
        #This is filled with the QN_Full_Loop function and is used to structure relevent pairs for superexchange
        #so that the program runs signifigantly faster.
        self.structured_pairs = []
        #Saves the angular eigenvalues calculated using the mathematica function. This avoids large redunant calculations.
        #These are used in the spin orbital coupling terms and can be indexed as Atom:orbital-orbital.
        #e.g. Er1->4f<1|-1> gets the spin orbital coupling <4f1|L|4f-1> and saves it as an [Lx, Ly, Lz]
        #list of contributions.
        self.angular_eigenvalues = {}
        #Initializes Direct Exchange directories.
        self.DE_ab = {}
        #This is a set of dictionaries to store useful sets of J sub-sumations
        self.J_ab = {}
        self.J_acdb = {}
        self.J_lalb = {}
        self.J_lalcldlb = {}
        self.J_lamcmdlb = {}
        self.J_malcldmb = {}
        self.J_mamcmdmb = {}
        #This handles the data for the AE tensor.
        self.AE_ab = {}
        self.AE_acdb = {}
        self.AE_lalb = {}
        self.AE_lalcldlb = {}
        self.AE_lamcmdlb = {}
        self.AE_malcldmb = {}
        self.AE_mamcmdmb = {}
        #This is a set of dictionaries to store useful sets of DM vectors. Since this is an antisymetric contribution
        #this set is the sum of forwards and backwards components. A different set gets the directional components.
        self.DM_ab_tot = {}
        self.DM_acdb_tot = {}
        self.DM_lalb_tot = {}
        self.DM_lalcldlb_tot = {}
        self.DM_lamcmdlb_tot = {}
        self.DM_malcldmb_tot = {}
        self.DM_mamcmdmb_tot = {}
        #This is a set of dictionaries to store useful sets of J sub-sumations
        self.DM_ab = {}
        self.DM_acdb = {}
        self.DM_lalb = {}
        self.DM_lalcldlb = {}
        self.DM_lamcmdlb = {}
        self.DM_malcldmb = {}
        self.DM_mamcmdmb = {}
        #This is a set of dictionaries to store useful sets of J sub-sumations
        self.DM_ba = {}
        self.DM_bcda = {}
        self.DM_lbla = {}
        self.DM_lblcldla = {}
        self.DM_lbmcmdla = {}
        self.DM_mblcldma = {}
        self.DM_mbmcmdma = {}
        #This is used to hold the strings that are later used for the function calls to calculate found t's.
        #These are basicly junk holders.
        self.t_Task_List = [['']*9,[]]
        self.key_dict = {}
        # Wolfram is initialized lazily so data inspection works without Mathematica.
        self.session = None
        
    #Clears any parameters needed for superexchange calculations.
    def Clear_Superexchange(self):
        self.angular_eigenvalues = {}
        self.found_t = {}
        self.charge_transfer_energies = {}
        #Initializes Direct Exchange directories.
        self.DE_ab = {}
        #This is a set of dictionaries to store useful sets of J sub-sumations
        self.J_ab = {}
        self.J_acdb = {}
        self.J_lalb = {}
        self.J_lalcldlb = {}
        self.J_lamcmdlb = {}
        self.J_malcldmb = {}
        self.J_mamcmdmb = {}
        #This handles the data for the AE tensor.
        self.AE_ab = {}
        self.AE_acdb = {}
        self.AE_lalb = {}
        self.AE_lalcldlb = {}
        self.AE_lamcmdlb = {}
        self.AE_malcldmb = {}
        self.AE_mamcmdmb = {}
        #This is a set of dictionaries to store useful sets of DM vectors. Since this is an antisymetric contribution
        #this set is the sum of forwards and backwards components. A different set gets the directional components.
        self.DM_ab_tot = {}
        self.DM_acdb_tot = {}
        self.DM_lalb_tot = {}
        self.DM_lalcldlb_tot = {}
        self.DM_lamcmdlb_tot = {}
        self.DM_malcldmb_tot = {}
        self.DM_mamcmdmb_tot = {}
        #This is a set of dictionaries to store useful sets of J sub-sumations
        self.DM_ab = {}
        self.DM_acdb = {}
        self.DM_lalb = {}
        self.DM_lalcldlb = {}
        self.DM_lamcmdlb = {}
        self.DM_malcldmb = {}
        self.DM_mamcmdmb = {}
        #This is a set of dictionaries to store useful sets of J sub-sumations
        self.DM_ba = {}
        self.DM_bcda = {}
        self.DM_lbla = {}
        self.DM_lblcldla = {}
        self.DM_lbmcmdla = {}
        self.DM_mblcldma = {}
        self.DM_mbmcmdma = {}
        self.t_Task_List = [['']*9,[]]
        self.key_dict = {}
        
    def Clear_Direct_Exchange(self):
        self.found_de = {}
        #Initializes Direct Exchange directories.
        self.DE_ab = {}
    
    #Used to load up all used mathematica functions. Note that his is basicly a mathematica document
    #written in python.
    def Init_Mathematica(self):
        if self.session is not None:
            return
        _require_wolfram()
        # If a Wolfram path is supplied, pass it through to the client.
        if self.wolfram_path in (None, "", "N/A"):
            self.session = WolframLanguageSession()
        else:
            self.session = WolframLanguageSession(self.wolfram_path)
        #Orbital Wave Functions
        self.session.evaluate('RadialOrbital[n_,l_,m_,Zeff_,x_,y_,z_]:=With[{r=Sqrt[x^2+y^2+z^2]},Sqrt[(((2*Zeff)/(n))^3)*((Factorial[n-l-1])/(2*n*(Factorial[n+l])))]*Exp[((-Zeff*r)/(n))]*(((2*Zeff*r)/(n))^l)*LaguerreL[n-l-1,2*l+1,(2*Zeff*r)/(n)]]//Simplify')
        self.session.evaluate('DisplacedRadialOrbital[n_,l_,m_,Zeff_,disvector_,x_,y_,z_]:=RadialOrbital[n,l,m,Zeff,x,y,z]/.{x->x-disvector[[1]],y->y-disvector[[2]],z->z-disvector[[3]]}')
        self.session.evaluate('AngularOrbital[n_,l_,m_,Zeff_,x_,y_,z_]:=With[{Nc=Sqrt[Factorial2[2l+1]/(4*Pi)],r=Sqrt[x^2+y^2+z^2]},Switch[l,0,Nc,1,Switch[m,0,Nc*z/r,-1,Nc*x/r,1,Nc*y/r],2,Switch[m,0,Nc*(3z^2-r^2)/(2*r^2*Sqrt[3]),-1,Nc*(x*z)/r^2,1,Nc*(y*z)/r^2,-2,Nc*(x*y)/r^2,2,Nc*(x^2-y^2)/(2r^2)],3,Switch[m,0,Nc*z(2z^2-3x^2-3y^2)/(2Sqrt[15]r^3),-1,Nc*x(4z^2-x^2-y^2)/(2Sqrt[10]r^3),1,Nc*y(4z^2-x^2-y^2)/(2Sqrt[10]r^3),-2,Nc*(x*y*z)/r^3,2,Nc*z(x^2-y^2)/(2r^3),-3,Nc*x(x^2-3y^2)/(2Sqrt[6]r^3),3,Nc*y(3x^2-y^2)/(2Sqrt[6]r^3)]]]')
        self.session.evaluate('DisplacedAngularOrbital[n_,l_,m_,Zeff_,disvector_,x_,y_,z_]:=AngularOrbital[n,l,m,Zeff,x,y,z]/.{x->x-disvector[[1]],y->y-disvector[[2]],z->z-disvector[[3]]}')
        self.session.evaluate('Orbital[n_,l_,m_,Zeff_,x_,y_,z_]:=RadialOrbital[n,l,m,Zeff,x,y,z]*AngularOrbital[n,l,m,Zeff,x,y,z]')
        self.session.evaluate('DisplacedOrbital[n_,l_,m_,Zeff_,disvector_,x_,y_,z_]:=DisplacedRadialOrbital[n,l,m,Zeff,disvector,x,y,z]*DisplacedAngularOrbital[n,l,m,Zeff,disvector,x,y,z]')
        #Kinetic Integral
        self.session.evaluate('KineticIntegral[n1_,l1_,m1_,Zeff1_,n2_,l2_,m2_,Zeff2_,disvector_]:=(-1/2)*Check[Quiet[NIntegrate[Simplify[Laplacian[Orbital[n1,l1,m1,Zeff1,x,y,z],{x,y,z}]]*DisplacedOrbital[n2,l2,m2,Zeff2,disvector,x,y,z],{x,-20,20},{y,-20,20},{z,-20,20}]],0]')
        #Potential Integral
        self.session.evaluate('NonDisplacedPotIntegral[n1_,l1_,m1_,Zeff1_,n2_,l2_,m2_,Zeff2_,disvector_]:=-Zeff1*Quiet[NIntegrate[Simplify[Orbital[n1,l1,m1,Zeff1,x,y,z]*DisplacedOrbital[n2,l2,m2,Zeff2,disvector,x,y,z]*(1/Sqrt[x^2+y^2+z^2])],{x,-20,20},{y,-20,20},{z,-20,20}]]')
        self.session.evaluate('DisplacedPotIntegral[n1_,l1_,m1_,Zeff1_,n2_,l2_,m2_,Zeff2_,disvector_]:=-Zeff2*Quiet[NIntegrate[Simplify[Orbital[n1,l1,m1,Zeff1,x,y,z]*DisplacedOrbital[n2,l2,m2,Zeff2,disvector,x,y,z]*(1/Sqrt[(x-disvector[[1]])^2+(y-disvector[[2]])^2+(z-disvector[[3]])^2])],{x,-20,20},{y,-20,20},{z,-20,20}]]')
        self.session.evaluate('PotIntegral[n1_,l1_,m1_,Zeff1_,n2_,l2_,m2_,Zeff2_,disvector_]:=Check[If[disvector=={0,0,0},NonDisplacedPotIntegral[n1,l1,m1,Zeff1,n2,l2,m2,Zeff2,disvector],NonDisplacedPotIntegral[n1,l1,m1,Zeff1,n2,l2,m2,Zeff2,disvector]+DisplacedPotIntegral[n1,l1,m1,Zeff1,n2,l2,m2,Zeff2,disvector]],0]')
        #Transfer Integral
        self.session.evaluate('TransferIntegral[n1_,l1_,m1_,Zeff1_,n2_,l2_,m2_,Zeff2_,disvector_]:=KineticIntegral[n1,l1,m1,Zeff1,n2,l2,m2,Zeff2,disvector]+PotIntegral[n1,l1,m1,Zeff1,n2,l2,m2,Zeff2,disvector]')
        #Angular Momentum Eigenvalues in Spherical Basis
        self.session.evaluate('AngXEigen[n1_,l1_,m1_,n2_,l2_,m2_]:=If[n2==n1,If[l2==l1,If[m2==m1-1,(1/2)*Sqrt[(l2*(l2+1))-(m2*(m2+1))],If[m2==m1+1,(1/2)*Sqrt[(l2*(l2+1))-(m2*(m2-1))],0]],0],0]')
        self.session.evaluate('AngYEigen[n1_,l1_,m1_,n2_,l2_,m2_]:=If[n2==n1,If[l2==l1,If[m2==m1-1,(-I/2)*Sqrt[(l2*(l2+1))-(m2*(m2+1))],If[m2==m1+1,(I/2)*Sqrt[(l2*(l2+1))-(m2*(m2-1))],0]],0],0]')
        self.session.evaluate('AngZEigen[n1_,l1_,m1_,n2_,l2_,m2_]:=If[n2==n1,If[l2==l1,If[m2==m1,m2,0],0],0]')
        #Angular Momentum in Cubic Basis
        self.session.evaluate('AngEigenNorm[n1_,l1_,m1_,n2_,l2_,m2_]:=(((I^Boole[(Mod[m1,2]==0&&m1<0)||(Mod[m1,2]==1&&m1>0)])/Sqrt[2])^Boole[m1!=0])*(((I^Boole[(Mod[m2,2]==0&&m2<0)||(Mod[m2,2]==1&&m2>0)])/Sqrt[2])^Boole[m2!=0])')
        self.session.evaluate('CubicAngXEigen[n1_,l1_,m1_,n2_,l2_,m2_]:=AngEigenNorm[n1,l1,m1,n2,l2,m2]*((Boole[m1!=0]*Boole[m2!=0]*AngXEigen[n1,l1,-Abs[m1],n2,l2,-Abs[m2]])+(((-1)^Boole[m2<0])*Boole[m1!=0]*AngXEigen[n1,l1,-Abs[m1],n2,l2,Abs[m2]])+(((-1)^Boole[m1<0])*Boole[m2!=0]*AngXEigen[n1,l1,Abs[m1],n2,l2,-Abs[m2]])+(((-1)^Boole[m1<0])*((-1)^Boole[m2<0])*AngXEigen[n1,l1,Abs[m1],n2,l2,Abs[m2]]))')
        self.session.evaluate('CubicAngYEigen[n1_,l1_,m1_,n2_,l2_,m2_]:=AngEigenNorm[n1,l1,m1,n2,l2,m2]*((Boole[m1!=0]*Boole[m2!=0]*AngYEigen[n1,l1,-Abs[m1],n2,l2,-Abs[m2]])+(((-1)^Boole[m2<0])*Boole[m1!=0]*AngYEigen[n1,l1,-Abs[m1],n2,l2,Abs[m2]])+(((-1)^Boole[m1<0])*Boole[m2!=0]*AngYEigen[n1,l1,Abs[m1],n2,l2,-Abs[m2]])+(((-1)^Boole[m1<0])*((-1)^Boole[m2<0])*AngYEigen[n1,l1,Abs[m1],n2,l2,Abs[m2]]))')
        self.session.evaluate('CubicAngZEigen[n1_,l1_,m1_,n2_,l2_,m2_]:=AngEigenNorm[n1,l1,m1,n2,l2,m2]*((Boole[m1!=0]*Boole[m2!=0]*AngZEigen[n1,l1,-Abs[m1],n2,l2,-Abs[m2]])+(((-1)^Boole[m2<0])*Boole[m1!=0]*AngZEigen[n1,l1,-Abs[m1],n2,l2,Abs[m2]])+(((-1)^Boole[m1<0])*Boole[m2!=0]*AngZEigen[n1,l1,Abs[m1],n2,l2,-Abs[m2]])+(((-1)^Boole[m1<0])*((-1)^Boole[m2<0])*AngZEigen[n1,l1,Abs[m1],n2,l2,Abs[m2]]))')
        self.session.evaluate('AngularVector[n1_,l1_,m1_,n2_,l2_,m2_]:={CubicAngXEigen[n1,l1,m1,n2,l2,m2],CubicAngYEigen[n1,l1,m1,n2,l2,m2],CubicAngZEigen[n1,l1,m1,n2,l2,m2]}//N')
        #Ionization Energy Function
        self.session.evaluate('EnergyExpectation[n_,l_,m_,Zeff_]:=TransferIntegral[n,l,m,Zeff,n,l,m,Zeff,{0,0,0}]')
        self.session.evaluate('OxidationEnergy[n_,l_,m_,Zeff_]:=EnergyExpectation[n,l,m,Zeff+0.35]-EnergyExpectation[n,l,m,Zeff]')
        self.session.evaluate('IonizationEnergy[n_,l_,m_,Zeff_]:=EnergyExpectation[n,l,m,Zeff-0.35]-EnergyExpectation[n,l,m,Zeff]')
        self.session.evaluate('ChargeTransferEnergy[n1_,l1_,m1_,Zeff1_,n2_,l2_,m2_,Zeff2_]:=OxidationEnergy[n1,l1,m1,Zeff1]+IonizationEnergy[n2,l2,m2,Zeff2]')
        #Overlap Integral
        self.session.evaluate('OverlapIntegral[n1_,l1_,m1_,Zeff1_,n2_,l2_,m2_,Zeff2_,disvector_]:=Quiet[NIntegrate[Simplify[Orbital[n1,l1,m1,Zeff1,x,y,z]*DisplacedOrbital[n2,l2,m2,Zeff2,disvector,x,y,z]],{x,-20,20},{y,-20,20},{z,-20,20}]]')
        #Exchange Integral
        self.session.evaluate("RadialExchangeIntegralR12[n1_,l1_,m1_,Zeff1_,n2_,l2_,m2_,Zeff2_,disvector_]:=(2/(2*l2+1))*Re[Integrate[DisplacedRadialOrbital[n1,l1,m1,Zeff1,-disvector,r2,theta2,phi2]*RadialOrbital[n2,l2,m2,Zeff2,r2]*(r2^(2+l2))*Integrate[RadialOrbital[n1,l1,m1,Zeff1,r1]*DisplacedRadialOrbital[n2,l2,m2,Zeff2,disvector,r1,theta1,phi1]*(r1^(1-l2)),{r1,r2,Infinity}],{r2,0,Infinity}]]")
        self.session.evaluate("CubicExchangeIntegral[n1_,l1_,m1_,Zeff1_,n2_,l2_,m2_,Zeff2_,disvector_]:=RadialExchangeIntegralR12[n1,l1,m1,Zeff1,n2,l2,m2,Zeff2,disvector]*((echarge^2)/(4*Pi*epsilon0))*KroneckerDelta[m1,m2,0]*2*Sqrt[Pi]")
        #Coulomb Integral
        self.session.evaluate("RadialCoulombIntegralR12[n1_,l1_,m1_,Zeff1_,n2_,l2_,m2_,Zeff2_,disvector_]:=Re[Integrate[(RadialOrbital[n2,l2,m2,Zeff2,r2]^2)*r2*Integrate[(RadialOrbital[n1,l1,m1,Zeff1,r1]^2)*(r1^2),{r1,0,r2}],{r2,0,Infinity}]]+Re[Integrate[(RadialOrbital[n2,l2,m2,Zeff2,r2]^2)*(r2^2)*Integrate[(RadialOrbital[n1,l1,m1,Zeff1,r1]^2)*r1,{r1,0,r2}],{r2,0,Infinity}]]")
        self.session.evaluate("CubicCoulombIntegral[n1_,l1_,m1_,Zeff1_,n2_,l2_,m2_,Zeff2_,disvector_]:=RadialCoulombIntegralR12[n1,l1,m1,Zeff1,n2,l2,m2,Zeff2,disvector]*((echarge^2)/(4*Pi*epsilon0))*KroneckerDelta[m1,m2,0]*2*Sqrt[Pi]")
        #Barbara Textbook Direct Exchange (Page 196)
        self.session.evaluate("CubicCoulombEnergy[n1_,l1_,m1_,Zeff1_,n2_,l2_,m2_,Zeff2_,disvector_]:=Integrate[CubicDisplacedOrbital[n1,l1,m1,Zeff1,disvector,r,theta,phi]^2*(Zeff2*echarge^2/(4*Pi*epsilon0))*r*Sin[theta],{r,0,Infinity},{theta,0,Pi},{phi,-Pi,Pi}]")
        self.session.evaluate("CubicBarbaraDirectExchange[n1_,l1_,m1_,Zeff1_,n2_,l2_,m2_,Zeff2_,disvector_]:=With[{OL=Re[CubicOverlapIntegral[n1,l1,m1,Zeff1,n2,l2,m2,Zeff2,disvector]]},If[Abs[OL]>0.001,-CubicExchangeIntegral[n1,l1,m1,Zeff1,n2,l2,m2,Zeff2,disvector]+(2*OL*CubicKineticIntegral[n1,l1,m1,Zeff1,n2,l2,m2,Zeff2,disvector])-((OL^2)*(CubicEnergyExpectationValue[n1,l1,m1,Zeff1]+CubicEnergyExpectationValue[n2,l2,m2,Zeff2]+CubicCoulombEnergy[n1,l1,m1,Zeff1,n2,l2,m2,Zeff2,disvector]+CubicCoulombEnergy[n2,l2,m2,Zeff2,n1,l1,m1,Zeff1,disvector]-CubicCoulombIntegral[n1,l1,m1,Zeff1,n2,l2,m2,Zeff2,disvector])),-CubicExchangeIntegral[n1,l1,m1,Zeff1,n2,l2,m2,Zeff2,disvector]]]")
    
    ######################################################################
    #Functions to get and show molecule information
    ######################################################################

    #Initializes or reinitializes the choses molecule.
    def get_Molecule_Data(self, file_name):
        # Keep track of the active data sheet and reset structured pairs if it changes.
        data_path = Path(file_name).expanduser()
        if file_name != self.data_sheet_name:
            self.data_sheet_name = file_name
            self.structured_pairs = []
        suffix = data_path.suffix.lower()
        if suffix == ".xlsx":
            self.molecule = pd.read_excel(data_path)
        elif suffix == ".csv":
            self.molecule = pd.read_csv(data_path)
        else:
            raise ValueError("Please use file type .csv or .xlsx")
    
    #Used to print current molecule panda array
    def show_Molecule(self):
        display(self.molecule)
    
    ######################################################################
    #Commands for indexing molecule panda array and organizing information
    ######################################################################
    
    #Just for readability. Takes in the molecule and the atom label and spits out the index in the list that
    #the molecule corresponds to for getting values. Note this fails if there are multiple atoms of the same label.
    def Atom_Index(self, atom):
        return self.molecule.index.get_loc(self.molecule[self.molecule["Atom"]==f"{atom}"].index[0])
    
    #Used to determine the class label that a given atom is associated with.
    def Class_Label(self, atom):
        return self.molecule["Class"][self.Atom_Index(atom)]
    
    #Used to determine the index in the class label list of a given class label.
    def Class_Label_Index(self, label):
        return self.molecule.index.get_loc(self.molecule[self.molecule["Class Labels"]==f"{label}"].index[0])
    
    #Used to determine the index in the class label list of a given atom's class.
    def Atom_Class_Label_Index(self, atom):
        return self.Class_Label_Index(self.Class_Label(atom))
    
    #Organizes an atom's coordinates into a list
    def get_Coords(self, atom):
        return [
            float(self.molecule["X"][self.Atom_Index(atom)]),
            float(self.molecule["Y"][self.Atom_Index(atom)]),
            float(self.molecule["Z"][self.Atom_Index(atom)]),
        ]
    
    #Gets the list of atoms bonded to the atom inquired on.
    def get_Bonds(self, atom):
        return [x.strip() for x in self.molecule["Bondings"][self.Atom_Index(atom)].split(",")]
    
    #Gets the Zeff for the given atom
    def Zeff(self, atom):
        return self.molecule["Zeff"][self.Atom_Class_Label_Index(atom)]
    
    #This gets the nuclear magneton for the given atom. Needed to calculate the dipole-dipole interaction.
    def BN(self, atom):
        return float(self.molecule["BN"][self.Atom_Class_Label_Index(atom)])
    
    #This gets the nuclear g factor for the given electron. Needed to calculate the dipole-dipole interaction.
    def gN(self, atom):
        return float(self.molecule["gN"][self.Atom_Class_Label_Index(atom)])
    ######################################################################
    #Functions for getting terms needed for calculation or indexing
    ######################################################################
    
    #Converts a list of cartesian coordinates to spherical coordinates.
    #Note, this gives devide by 0 if cart_coords = [0, 0, 0]
    def get_Spher_Coords(self, cart_coords):
        if cart_coords[0] == 0:
             return [np.sqrt(cart_coords[0]**2 + cart_coords[1]**2 + cart_coords[2]**2), np.pi/2, np.arccos(cart_coords[2]/
                                                                (np.sqrt(cart_coords[0]**2 + d[1]**2 + d[2]**2)))]
        else:
            return [np.sqrt(cart_coords[0]**2 + cart_coords[1]**2 + cart_coords[2]**2), np.arctan(cart_coords[1]/(cart_coords[0])), 
                np.arccos(cart_coords[2]/(np.sqrt(cart_coords[0]**2 + cart_coords[1]**2 + cart_coords[2]**2)))]
    
    #Gets displacement vector between 2 atoms
    def get_Displacement(self, atom1_from, atom2_to):
        cart_atom1 = self.get_Coords(atom1_from)
        cart_atom2 = self.get_Coords(atom2_to)
        return [cart_atom2[0] - cart_atom1[0], cart_atom2[1] - cart_atom1[1], cart_atom2[2] - cart_atom1[2]]
    
    #Gets displacement vector in spherical coordinates.
    def get_Spher_Displacement(self, atom1_from, atom2_to):
        return self.get_Spher_Coords(self.get_Displacement(atom1_from, atom2_to))
    
    #Used to get the list of orbitals used in exchange for a given atom.
    def Orbitals(self, atom):
        return [x.strip() for x in self.molecule["Orbitals"][self.Atom_Class_Label_Index(atom)].split(",")]
    
    #Tells you if the atom is a class for starting or ending the exchange chain.
    def is_Start_End(self, atom):
        if self.molecule["Exchange Atom"][self.Atom_Class_Label_Index(atom)].split(", ")[0] in ["Y", "y", "T", "t"]:
            return True
        else:
            return False
    
    #This gets the index of an orbital in the "Orbital"
    def get_Orbital_Index(self, atom, orbital):
        return self.Orbitals(atom).index(f"{orbital}")
    
    #Gets the total spin of an orbital in an atom.
    def Atom_Spin(self, atom):
        return float(self.molecule["Spin"][self.Atom_Class_Label_Index(atom)])
    
    #Gets the spin orbital coupling of an atom.
    def SOCC(self, atom):
        return float(self.molecule["SOCC"][self.Atom_Class_Label_Index(atom)])
    
    #Returns the list of energy levels for various m values of a given orbital for a specific atom
    #Values should be determined by DFT and account for crystaline field splitting!
    def get_Crystal_Field_Energies(self, atom, orbital):
        return list(map(float, self.molecule["Crystal Field Splitting"][self.Atom_Index(atom)].split("|"
                                                                    )[self.get_Orbital_Index(atom, orbital)].split(",")))
    
    #Likely want to redefine the core function to this
    def get_Quantum_Numbers(self, orbital):
        if len(orbital) != 2:
            #Returns n, l, m
            q_nums = [int(orbital[0]), str.find("spdfghij", orbital[1]), int(orbital[2:])]
            if type(q_nums[1]) != int or type(q_nums[0]) != int or q_nums[1] < 0 or q_nums[1] > q_nums[0]-1:
                sys.exit(f"Quantum numbers must be integers where 0 <= l < n, but l = {q_nums[1]} and n = {q_nums[0]}")
            if type(q_nums[2]) != int or q_nums[1] < np.abs(q_nums[2]):
                sys.exit(f"Quantum numbers must be integers and l >= |m|, but l = {q_nums[1]} and m = {q_nums[2]}")
            return q_nums
        else:
            #Returns n, l, m where m is al list of all the possible m's for the given l's
            l = str.find("spdfghij", orbital[1])
            q_nums = [int(orbital[0]), l, list(range(-l, l+1))]
            if type(q_nums[1]) != int or type(q_nums[0]) != int or q_nums[1] < 0 or q_nums[1] > q_nums[0]-1:
                sys.exit(f"Quantum numbers must be integers where 0 <= l < n, but l = {q_nums[1]} and n = {q_nums[0]}")
            return q_nums
    
    #Retreives orbital label from set of quantum numbers. Useful for organizing labels for data.  
    def get_Orbital_Label(self, n, l, m = ""):
        if type(l) != int or type(n) != int or l < 0 or l > n-1:
            sys.exit(f"Quantum numbers must be integers where 0 <= l < n, but l = {l} and n = {n}")
        if type(m) == int:
            if l < np.abs(m):
                sys.exit(f"Quantum numbers must be integers and l >= |m|, but l = {l} and m = {m}")
            return f"{n}{'spdfghij'[l]}{m}"
        else:
            return f"{n}{'spdfghij'[l]}"
    
    #This is used to get a specific energy of a set of quantum numbers in a specific atom in the lattice.
    def get_Orbital_Energy(self, atom, orbital, m):
        n, l, m_nah = self.get_Quantum_Numbers(f"{orbital}")
        if np.abs(m) <= l:
            return self.get_Crystal_Field_Energies(atom, orbital)[m+l]
        else:
            print(f"For get_Orbital_Energy({atom},{orbital},{m}), |m| = {np.abs(m)} > l = {l}. This is not allowed.")
            print(f"Recommended fix: Check the orbital energy levels listed in {atom} and ensure the order matches the orbitals in {self.Class_Label(atom)} in the excel sheet.")
            sys.exit(-1)
            
    #Calculates the transfer integral for 2 atoms. Kept seperate from t() for readability.
    def calc_t(self, atom_from, orbital_from, atom_to, orbital_to, command_string = False):
        n1, l1, m1 = self.get_Quantum_Numbers(orbital_from)
        n2, l2, m2 = self.get_Quantum_Numbers(orbital_to)
        d = self.get_Displacement(atom_from, atom_to)
        vec_d = "{" + f"{d[0]},{d[1]},{d[2]}" + "}"
        if command_string:
            return [str(n1),str(l1),str(m1),str(self.Zeff(atom_from)),str(n2),str(l2),str(m2),str(self.Zeff(atom_to)),vec_d]
        else:
            return self.session.evaluate(f'TransferIntegral[{n1},{l1},{m1},{self.Zeff(atom_from)},{n2},{l2},{m2},{self.Zeff(atom_to)},{vec_d}]')
    
    #Calculates the direct exchange integral for 2 atoms. Kept seperate from de() for readability.
    def calc_de(self, atom_from, orbital_from, atom_to, orbital_to, command_string=False):
        n1, l1, m1 = self.get_Quantum_Numbers(orbital_from)
        n2, l2, m2 = self.get_Quantum_Numbers(orbital_to)
        d = self.get_Displacement(atom_from, atom_to)
        vec_d = "{" + f"{d[0]},{d[1]},{d[2]}" + "}"
        if command_string:
            return [str(n1), str(l1), str(m1), str(self.Zeff(atom_from)), str(n2), str(l2), str(m2),
                    str(self.Zeff(atom_to)), vec_d]
        else:
            return self.session.evaluate(f'DirectExchange[{n1},{l1},{m1},{self.Zeff(atom_from)},{n2},{l2},{m2},{self.Zeff(atom_to)},{vec_d}]')
    
    #Gets the vector norm of 2 atoms.
    def Vec_Norm(self, xyz_1, xyz_2):
        return np.sqrt((xyz_1[0] - xyz_2[0])**2 + (xyz_1[1] - xyz_2[1])**2 + (xyz_1[2] - xyz_2[2])**2)

    #Gets the bond angle, where atom_list is a 3-length list of atom labels. The centeral atom is the center index of the angle.
    #Note this means [O1, O2, O3] = [O3, O2, O1]
    #This works by making use of the basic cosine rule idea from geometry to solve for an angle of a triangle when knowing
    #only the lengths of the sides. Note that the vector norm of the difference of coordinates as vectors
    #forms a triangle within the plane enhabited by all 3 atom's spacial coordinates.
    #The triangle has lengths equal to the respective vector norm of the differece of the 2 connecting points for any given
    #side, so we can calculate the angle using cosine rule and these norms.
    #Note: By default, the angle is calculated in radians. If you want degrees, enter "deg" in the 3rd entry.
    #Typos/left blank go to radians automatically.
    def get_Bond_Angle(self, atom_list, deg_rad = "rad"):
        #This allows us to either take in a list of 3 elements, [x,y,z], for an atom or the atom label to get this
        #this form the sheet.
        if type(atom_list[0]) == str:
            atom_list[0] = self.get_Coords(atom_list[0])
        if type(atom_list[1]) == str:
            atom_list[1] = self.get_Coords(atom_list[1])
        if type(atom_list[2]) == str:
            atom_list[2] = self.get_Coords(atom_list[2])
        a = self.Vec_Norm(atom_list[0], atom_list[1])
        b = self.Vec_Norm(atom_list[1], atom_list[2])
        c = self.Vec_Norm(atom_list[0], atom_list[2])
        if deg_rad == "deg":
            deg_rad_mod = 180/np.pi
        else:
            deg_rad_mod = 1
        return deg_rad_mod*math.acos((a**2 + b**2 - c**2)/(2*a*b)) #Gives angle in radians by default.

    #Calculates all bond angles in the molecule.
    #Note: By default, angle is in radians. If you want degrees, enter "deg" in the 3rd entry.
    #Typos/left blank go to radians automatically.
    def get_All_Bond_Angles(self, deg_rad = "deg"):
        if deg_rad == "rad":
            unit = "radians"
        else:
            unit = "degrees"
        bond_angle_list = {"Bond":[], f"Angle({unit})":[]}
        for atom in self.molecule["Atom"]:
            #Adds new directory for all bond angles centered around this atom.
            #Retreives the bondings string and converts it to a list of the atom's bonded atoms by their label.
            bonding_list = self.get_Bonds(atom)
            if len(bonding_list) > 1: #If <1, no angles exist with this atom at the center, so we skip it.
                #Calculates the angles of any bond pairs where the atom is at the center.
                for i in list(range(len(bonding_list))):
                    #Stops the loop at the last atom, since it will never have any unique 3rd atom.
                    if i == len(bonding_list) - 1:
                        break
                    #Loops through all remaining bonding angle pairs for the current atom and calculates them
                    for j in list(range(len(bonding_list)))[i+1:]:
                        bond_angle_list["Bond"] += [f"{bonding_list[i]}-{atom}-{bonding_list[j]}"]
                        bond_angle_list[f"Angle({unit})"] += [
                            self.get_Bond_Angle([bonding_list[i], atom, bonding_list[j]], deg_rad)
                        ]
        return pd.DataFrame.from_dict(bond_angle_list)

    #######################################################################################
    #Takes in orbital from/to in the usual form (e.g. 3p-1, 4f3) and mom_lab is the momentum label 0=x, 1=y, 2=z, 3 = all
    def Exp_Moment(self, orbital_from, orbital_to, mom_lab):
        n1, l1, m1 = self.get_Quantum_Numbers(orbital_from)
        n2, l2, m2 = self.get_Quantum_Numbers(orbital_to)
        if m1 == m2 and mom_lab in [2]:
            return m1
        elif np.abs(m1 - m2) == 1 and mom_lab in [0, 1]:
            return 0.5*np.sqrt(l1*(l1+1)-m1*(m1+(m2-m1)*l1))*((1.0j)**mom_lab)
        elif mom_lab == 3:
            return ((1.0j)**mom_lab)*(0.5*np.sqrt(l1*(l1+1)-m1*(m1+(m2-m1)*l1)))*(np.abs(m1-m2) == 1) + m1*(m1==m2)
        elif (mom_lab in [0, 1, 2, 3]) == False:
            print("Momentum label must be either 0 for x, 1 for y, 2 for z, or 3 for the sum of all 3.")
            sys.exit(-1)
        else:
            return 0
            
    #Used to either pull or calculate the transfer integrals.
    #Note that if only an orbital indicating n and l is input, this will calculate all
    def t(self, atom_from, orbital_from, atom_to, orbital_to):
        #Checks if this has already been calculated and pulls the value if we have.
        if f"{atom_from}({orbital_from})-{atom_to}({orbital_to})" in self.found_t:
            return self.found_t[f"{atom_from}({orbital_from})-{atom_to}({orbital_to})"]
        #Determines the list of m1 values, depending on if we care about only 1 value of m values or all combinations.
        print(f"Attempted to recalculate: {f'{atom_from}({orbital_from})-{atom_to}({orbital_to})'}")
        if len(orbital_from) == 2:
            n1, l1 = self.get_Quantum_Numbers(orbital_from)
            m1 = range(-l1, l1+1)
        else:
            m1 = [0]
            n1, l1, m1[0] = self.get_Quantum_Numbers(orbital_from)
        #Determines the list of m2 values, depending on if we care about only 1 value of m2 values or all combinations.
        if len(orbital_to) == 2:
            n2, l2 = self.get_Quantum_Numbers(orbital_to)
            m2 = range(-l2, l2+1)
        else:
            m2 = [0]
            n2, l2, m2[0] = self.get_Quantum_Numbers(orbital_to)
        found_t_val = 0
        for i in m1:
            orbital1 = self.get_Orbital_Label(n1,l1,i)
            for j in m2:
                orbital2 = self.get_Orbital_Label(n2,l2,j)
                self.found_t[f"{atom_from}({orbital1})-{atom_to}({orbital2})"] = self.calc_t(atom_from, orbital1, atom_to, orbital2)
                found_t_val += self.found_t[f"{atom_from}({orbital1})-{atom_to}({orbital2})"]
        self.found_t[f"{atom_from}({orbital_from})-{atom_to}({orbital_to})"] = found_t_val
        return self.found_t[f"{atom_from}({orbital1})-{atom_to}({orbital2})"]
    
    # Pull or calculate direct exchange integrals
    # Pull or calculate direct exchange integrals
    def de(self, atom_from, orbital_from, atom_to, orbital_to):
        # Checks if this has already been calculated and pulls the value if we have.
        if f"{atom_from}({orbital_from})-{atom_to}({orbital_to})" in self.found_de:
            return self.found_de[f"{atom_from}({orbital_from})-{atom_to}({orbital_to})"]
        # Determines the list of m1 values, depending on if we care about only 1 value of m values or all combinations.
        if len(orbital_from) == 2:
            n1, l1 = self.get_Quantum_Numbers(orbital_from)
            m1 = list(range(-l1, l1 + 1))
        else:
            m1 = [0]
            n1, l1, m1[0] = self.get_Quantum_Numbers(orbital_from)
        # Determines the list of m2 values, depending on if we care about only 1 value of m2 values or all combinations.
        if len(orbital_to) == 2:
            n2, l2 = self.get_Quantum_Numbers(orbital_to)
            m2 = list(range(-l2, l2 + 1))
        else:
            m2 = [0]
            n2, l2, m2[0] = self.get_Quantum_Numbers(orbital_to)
        found_t_val = 0
        for i in m1:
            orbital1 = self.get_Orbital_Label(n1, l1, i)
            for j in m2:
                orbital2 = self.get_Orbital_Label(n2, l2, j)
                self.found_de[f"{atom_from}({orbital1})-{atom_to}({orbital2})"] = self.calc_de(atom_from, orbital1,
                                                                                               atom_to, orbital2)
                found_t_val += self.found_de[f"{atom_from}({orbital1})-{atom_to}({orbital2})"]
        self.found_de[f"{atom_from}({orbital_from})-{atom_to}({orbital_to})"] = found_t_val
        return self.found_de[f"{atom_from}({orbital1})-{atom_to}({orbital2})"]
    
    #Gets the charge transfer energy between who atoms
    def Charge_Transfer(self, atom_from, atom_to):
        atom_from_class = self.Class_Label(atom_from)
        atom_to_class = self.Class_Label(atom_to)
        if f"{atom_from_class}-{atom_to_class}" in self.charge_transfer_energies:
            delta = self.charge_transfer_energies[f"{atom_from_class}-{atom_to_class}"]
            return delta
        else:
            n1, l1, m1 = self.get_Quantum_Numbers(self.Orbitals(atom_from)[0])
            n2, l2, m2 = self.get_Quantum_Numbers(self.Orbitals(atom_to)[0])
            self.Init_Mathematica()
            self.charge_transfer_energies[f"{atom_from_class}-{atom_to_class}"] = self.session.evaluate(f"ChargeTransferEnergy[{n1},{l1},0,{self.Zeff(atom_from)},{n2},{l2},0,{self.Zeff(atom_to)}]")
            delta = self.charge_transfer_energies[f"{atom_from_class}-{atom_to_class}"]
            return delta
 
    #Retreives or calculates the angular expectation for spin orbital coupling over 2 m quantum numbers
    #in a given orbital [n, l values].
    def Angular_Expect(self, atom, n, l, m_bra, m_ket):
        if f"{atom}->{n}{'spdfghi'[l]}<{m_bra}|{m_ket}>" in self.angular_eigenvalues:
            return self.angular_eigenvalues[f"{atom}->{n}{'spdfghi'[l]}<{m_bra}|{m_ket}>"]
        else:
            self.Init_Mathematica()
            #list_term = list(self.session.evaluate(f"AngularVector[{n},{l},{m_bra}, {n},{l},{m_ket}]"))
            list_term = list(self.session.evaluate(f"AngularVector[{n},{l},{m_bra},{n},{l},{m_ket}]"))
            #Comment below is if python functions are used
            '''list_term = list(self.cubicangulareigen(n,l,m_bra,n,l,m_ket))'''
            #Organizes the list into python formatting
            conjugate_term = [0,0,0]
            for i in range(len(list_term)):
                #This indicates that it is a complex number rather then a float. We want to put this in python's
                #notation for complex numbers.
                if str(list_term[i])[0] == "C":
                    num_list = str(list_term[i]).split("[")[1].split(",")
                    list_term[i] = float(num_list[0]) + float(num_list[1][:-1])*1j
                    conjugate_term[i] = float(num_list[0]) - float(num_list[1][:-1])*1j
                else:
                    conjugate_term[i] = list_term[i]
                    #Comment below is if python functions are used
                    '''conjugate_term[i] = -list_term[i]'''
            self.angular_eigenvalues[f"{atom}->{n}{'spdfghi'[l]}<{m_bra}|{m_ket}>"] = list_term
            self.angular_eigenvalues[f"{atom}->{n}{'spdfghi'[l]}<{m_ket}|{m_bra}>"] = conjugate_term
            return self.angular_eigenvalues[f"{atom}->{n}{'spdfghi'[l]}<{m_bra}|{m_ket}>"]

    #########################################
    #Internal deffinitions are made for the mapping function to increase speed. Once called, it will organize everything
    #for later functions to use in the saved variable "structured_pairs"
    def QN_Full_Loop(self):
        ab_pair = list(it.product(self.get_Start_End_Atoms(), self.get_Start_End_Atoms()))

        def acdb_arrange(ab, cd):
            #Returns none for filtering if it has already been evaluated from the other direction to not duplicate work.
            if f"{ab[1]}-{cd[0]}-{cd[1]}-{ab[0]}" not in self.J_acdb:
                self.J_acdb[f"{ab[0]}-{cd[0]}-{cd[1]}-{ab[1]}"] = 0
                #Add the rest if this works.
                return ab + cd

        def uniq_pairing(ab):
        #Checks if the reverse has already been calculated.
            if f"{ab[1]}-{ab[0]}" not in self.J_ab:
                shared_bond = self.get_Shared_Bonds(ab[0],ab[1])
                if ab[0] in shared_bond:
                    shared_bond.remove(ab[0])
                elif ab[1] in shared_bond:
                    shared_bond.remove(ab[1])
                #If there is no share bonds, there is no superexchange, so we skip that one. Early simplification.
                if shared_bond != []:
                    self.J_ab[f"{ab[0]}-{ab[1]}"] = 0
                    #add all the rest of the initializers if this works here.
                    shared_bond_product = list(it.product(shared_bond, repeat = 2))
                    acdb_list = list(filter(None, list(map(acdb_arrange, [ab]*len(shared_bond_product), shared_bond_product))))
                    return acdb_list

        #Gets and reformats the information for later deffinitions.
        acdb_pair = list(filter(None, map(uniq_pairing, it.product(self.get_Start_End_Atoms(), self.get_Start_End_Atoms()))))
        #Quick reformatting
        acdb = []
        for acdb_set in acdb_pair:
            acdb += acdb_set

        #Used to format the quantum numbers in a mapping function from something like 1, 1, [-1,0,1] to  [[1,1,-1],[1,1,0],[1,1,1]] 
        def Segment_Quantum_Numbers(n, l, m):
            return [n, l, m]

        #More reorganizing definitions
        def Atom_QN_Map(a, qn):
            return [a, qn[0], qn[1], qn[2]]

        def Quantum_Number_Sets(a, b, c, d,o_abcd):
            #Used to remove cases where you start and end at the same atom in the same orbital.
            if a != b or o_abcd[0][0] != o_abcd[0][1]:
                #used to avoid double counting/alculating since backwards paths are already accounted for in the equations.
                #Returns none if it's already been accounted for and doesn't do the calculation, allowing for faster
                #run time and easy filtering.
                is_acadlcld = (c != d) or (o_abcd[1][0] != o_abcd[1][1])
                if f"{b}({o_abcd[0][1]})-{c}({o_abcd[1][0]})" + f"-{d}({o_abcd[1][1]})"*is_acadlcld + f"-{a}({o_abcd[0][0]})" not in self.J_lalcldlb:
                    self.J_lalcldlb[f"{a}({o_abcd[0][0]})-{c}({o_abcd[1][0]})" + f"-{d}({o_abcd[1][1]})"*is_acadlcld + f"-{b}({o_abcd[0][1]})"] = 0
                    na, la, ma = self.get_Quantum_Numbers(o_abcd[0][0])
                    nb, lb, mb = self.get_Quantum_Numbers(o_abcd[0][1])
                    qn_a = list(map(Segment_Quantum_Numbers, [na]*len(ma), [la]*len(ma), ma))
                    qn_b = list(map(Segment_Quantum_Numbers, [nb]*len(mb), [lb]*len(mb), mb))
                    nc, lc, mc = self.get_Quantum_Numbers(o_abcd[1][0])
                    qn_c = list(map(Segment_Quantum_Numbers, [nc]*len(mc), [lc]*len(mc), mc))
                    nd, ld, md = self.get_Quantum_Numbers(o_abcd[1][1])
                    qn_d = list(map(Segment_Quantum_Numbers, [nd]*len(md), [ld]*len(md), md))
                    return list(it.product(list(map(Atom_QN_Map,[a]*len(qn_a), qn_a)), list(map(Atom_QN_Map,[b]*len(qn_b), qn_b)), 
                                       list(map(Atom_QN_Map,[c]*len(qn_c), qn_c)), list(map(Atom_QN_Map,[d]*len(qn_d), qn_d))))

        def get_Atom_Quantum_Number_Pairs(abcd):
            o_a = self.Orbitals(abcd[0])
            o_b = self.Orbitals(abcd[1])
            o_c = self.Orbitals(abcd[2])
            o_d = self.Orbitals(abcd[3])
            o_acdb = list(it.product(it.product(o_a, o_b), it.product(o_c, o_d)))
            o_acdb_len = len(o_acdb)
            return list(filter(None, map(Quantum_Number_Sets, [abcd[0]]*o_acdb_len, [abcd[1]]*o_acdb_len, [abcd[2]]*o_acdb_len, [abcd[3]]*o_acdb_len, o_acdb)))
        
        x = list(map(get_Atom_Quantum_Number_Pairs, acdb))#get_Atom_Quantum_Number_Pairs("Er1", "Er2", "O1", "O2"))
        #Takes advantage of list grouplings to quickly organize final set! Small enough double for loop is fine!
        self.structured_pairs = []
        for acdb_set in x:
            for subset in acdb_set:
                self.structured_pairs += subset
        
    #This is simply here so this can be used as the functions for mapping to combine lists faster.
    def Add_List_Val(self, val1, val2):
        return val1+val2
    
    #This gets the energy difference between 2 states. If it's the same atom and orbital, the difference is due to
    #due to crystaline field splitting. Note that tis is Delta_12 = E2-E1
    def Delta_12(self, atom1, orbital1, m1, atom2, orbital2, m2):
        crystal_energy = (self.get_Orbital_Energy(atom2, orbital2, m2) - self.get_Orbital_Energy(atom1, orbital1, m1))
        return crystal_energy
        
    #Set of deffinitions Used to calculate Symetric Isotropic Component.
    def get_Start_End_Atoms(self):
        atom_list = list(self.molecule["Atom"])
        exchange_list = []
        for atom in atom_list:
            if self.is_Start_End(atom):
                exchange_list += [atom]
        return exchange_list
    
    #Gets the list of shared bonds between 2 atoms.
    def get_Shared_Bonds(self, atom1, atom2):
        bond_list_1 = self.get_Bonds(atom1)
        bond_list_2 = self.get_Bonds(atom2)
        shared_bonds = []
        for bond1 in bond_list_1:
            if bond1 in bond_list_2:
                shared_bonds += [bond1]
        return shared_bonds
    
    #This organizes the string needed for calculating the found_t along with minimizing the number of calculated t.
    def Organize_Directories(self, abcd):
        #Gets the information needed to make the keys for this.
        a_a, na, la, ma = abcd[0]
        o_a = f'{na}{"spdfghij"[la]}'
        a_b, nb, lb, mb = abcd[1]
        o_b = f'{nb}{"spdfghij"[lb]}'
        a_c, nc, lc, mc = abcd[2]
        o_c = f'{nc}{"spdfghij"[lc]}'
        a_d, nd, ld, md = abcd[3]
        o_d = f'{nd}{"spdfghij"[ld]}'
        #Simply used to condence if statements for faster processing times and denser code when assigning keys.
        is_acadocod = ((a_c != a_d) or (o_c != o_d))
        is_acadocodmcmd = ((is_acadocod) or (mc != md))
        
        #Now we can organize the directories. Note that the 
        mamcmdmb_key = f"{a_a}({o_a}{ma})-{a_c}({o_c}{mc})" + f"-{a_d}({o_d}{md})"*is_acadocodmcmd + f"-{a_b}({o_b}{mb})"
        self.J_mamcmdmb[mamcmdmb_key] = 0
        self.AE_mamcmdmb[mamcmdmb_key] = [0]*9
        if a_c == a_d:
            self.DM_mamcmdmb_tot[mamcmdmb_key] = [0]*9
            self.DM_mamcmdmb[mamcmdmb_key] = [0]*3
            self.DM_mbmcmdma[mamcmdmb_key] = [0]*3
        
        lamcmdlb_tag = f"{a_a}({o_a})-{a_c}({o_c}{mc})" + f"-{a_d}({o_d}{md})"*is_acadocodmcmd + f"-{a_b}({o_b})"
        if lamcmdlb_tag not in self.AE_lamcmdlb:
            self.J_lamcmdlb[lamcmdlb_tag] = 0
            self.AE_lamcmdlb[lamcmdlb_tag] = [0]*9
            if a_c == a_d:
                self.DM_lamcmdlb_tot[lamcmdlb_tag] = [0]*3
                self.DM_lamcmdlb[lamcmdlb_tag] = [0]*3
                self.DM_lbmcmdla[lamcmdlb_tag] = [0]*3
            
        malcldmb_tag = f"{a_a}({o_a}{ma})-{a_c}({o_c})" + f"-{a_d}({o_d})"*is_acadocod + f"-{a_b}({o_b}{mb})"
        if malcldmb_tag not in self.AE_malcldmb:
            self.J_malcldmb[malcldmb_tag] = 0
            self.AE_malcldmb[malcldmb_tag] = [0]*9
            if a_c == a_d:
                self.DM_malcldmb_tot[malcldmb_tag] = [0]*3
                self.DM_malcldmb[malcldmb_tag] = [0]*3
                self.DM_mblcldma[malcldmb_tag] = [0]*3
        
        #AE without J because this has already been built in J.
        lalcldlb_tag = f"{a_a}({o_a})-{a_c}({o_c})" + f"-{a_d}({o_d})"*is_acadocod + f"-{a_b}({o_b})"
        if lalcldlb_tag not in self.AE_lalcldlb:
            self.AE_lalcldlb[lalcldlb_tag] = [0,0,0,0,0,0,0,0,0]
            if a_c == a_d:
                self.DM_lalcldlb_tot[lalcldlb_tag] = [0]*3
                self.DM_lalcldlb[lalcldlb_tag] = [0]*3
                self.DM_lblcldla[lalcldlb_tag] = [0]*3
            
        lalb_tag = f"{a_a}({o_a})-{a_b}({o_b})"
        if lalb_tag not in self.AE_lalb:
            self.J_lalb[lalb_tag] = 0
            self.AE_lalb[lalb_tag] = [0]*9
            if a_c == a_d:
                self.DM_lalb_tot[lalb_tag] = [0]*3
                self.DM_lalb[lalb_tag] = [0]*3
                self.DM_lbla[lalb_tag] = [0]*3
        
        acdb_tag = f"{a_a}-{a_c}" + f"-{a_d}"*(a_c != a_d) + f"-{a_b}"
        if acdb_tag not in self.AE_acdb:
            self.J_acdb[acdb_tag] = 0
            self.AE_acdb[acdb_tag] = [0]*9
            if a_c == a_d:
                self.DM_acdb_tot[acdb_tag] = [0]*3
                self.DM_acdb[acdb_tag] = [0]*3
                self.DM_bcda[acdb_tag] = [0]*3
        
        #AE without J because this has already been built in J.
        if f"{a_a}-{a_b}" not in self.AE_ab:
            self.AE_ab[f"{a_a}-{a_b}"] = [0]*9
            if a_c == a_d:
                self.DM_ab_tot[f"{a_a}-{a_b}"] = [0]*3
                self.DM_ab[f"{a_a}-{a_b}"] = [0]*3
                self.DM_ba[f"{a_a}-{a_b}"] = [0]*3
                
        #Also organizes and optimizes against redundancies for the transfer integral calculations.
        if f"{a_c}({o_c}{mc})-{a_b}({o_b}{mb})" not in self.key_dict:
            self.t_Task_List[1] = self.t_Task_List[1] + [f"{a_c}({o_c}{mc})-{a_b}({o_b}{mb})"]
            self.key_dict[f"{a_c}({o_c}{mc})-{a_b}({o_b}{mb})"] = 1
            next_param = self.calc_t(a_c,f'{o_c}{mc}',a_b,f'{o_b}{mb}', True)
            for i in range(9):
                self.t_Task_List[0][i] += next_param[i] + ","

        if f"{a_a}({o_a}{ma})-{a_c}({o_c}{mc})" not in self.key_dict:
            self.t_Task_List[1] = self.t_Task_List[1] + [f"{a_a}({o_a}{ma})-{a_c}({o_c}{mc})"]
            self.key_dict[f"{a_a}({o_a}{ma})-{a_c}({o_c}{mc})"] = 1
            next_param = self.calc_t(a_a,f'{o_a}{ma}',a_c,f'{o_c}{mc}', True)
            for i in range(9):
                self.t_Task_List[0][i] += next_param[i] + ","

        if f"{a_d}({o_d}{md})-{a_a}({o_a}{ma})" not in self.key_dict:
            self.t_Task_List[1] = self.t_Task_List[1] + [f"{a_d}({o_d}{md})-{a_a}({o_a}{ma})"]
            self.key_dict[f"{a_d}({o_d}{md})-{a_a}({o_a}{ma})"] = 1
            next_param = self.calc_t(a_d,f'{o_d}{md}',a_a,f'{o_a}{ma}', True)
            for i in range(9):
                self.t_Task_List[0][i] += next_param[i] + ","

        if f"{a_b}({o_b}{mb})-{a_d}({o_d}{md})" not in self.key_dict:
            self.t_Task_List[1] = self.t_Task_List[1] + [f"{a_b}({o_b}{mb})-{a_d}({o_d}{md})"]
            self.key_dict[f"{a_b}({o_b}{mb})-{a_d}({o_d}{md})"] = 1
            next_param = self.calc_t(a_b,f'{o_b}{mb}',a_d,f'{o_d}{md}', True)
            for i in range(9):
                self.t_Task_List[0][i] += next_param[i] + ","
    
    #This is to calculate all t using the new structure:
    def Calculate_All_t(self):
        self.Init_Mathematica()
        #Gets all the pairs from the structure specified by the molecule sheet and saves them to the structured_pairs list.
        self.QN_Full_Loop()
        #map(self.Organize_Directories, self.structured_pairs)
        #Uses for loop at this stage since map won't save to the variables within the class. Slightly slower but still
        #a large improvement.
        for structured_pair in self.structured_pairs:
            self.Organize_Directories(structured_pair)
        #Now that we determine all the tasks, we want to use parallel_evaluate to calculate this faster by opening multiple
        #kernals. In theory, this should decrease the processing time by an inverse factor of the number of kernals,
        #so using a supercomputer should make this extremely fast!
        #param = Transpose[{}]
        #Organizes the string to input. Aweful looking but fast enough relative to the actual calculation.
        eval_string = f'{"{{"+self.t_Task_List[0][0][:-1]+"}"},{"{"+self.t_Task_List[0][1][:-1]+"}"},{"{"+self.t_Task_List[0][2][:-1]+"}"},{"{"+self.t_Task_List[0][3][:-1]+"}"},{"{"+self.t_Task_List[0][4][:-1]+"}"},{"{"+self.t_Task_List[0][5][:-1]+"}"},{"{"+self.t_Task_List[0][6][:-1]+"}"},{"{"+self.t_Task_List[0][7][:-1]+"}"},{"{"+self.t_Task_List[0][8][:-1]+"}}"}'
        t_values = self.session.evaluate(f'ParallelMap[Apply[TransferIntegral], Transpose[{eval_string}]]')
        self.t_values = t_values
        #This shuts down the extra sessions then reloads the session for our future components
        #that don't need to work in parallel.
        self.session.terminate()
        self.session = None
        self.Init_Mathematica()
        for i in range(len(t_values)):
            self.found_t[self.t_Task_List[1][i]] = float(t_values[i])

    #Calcultes all the direct exchanges with the end atoms.
    def Calculate_All_de(self):
        self.Init_Mathematica()
        exchange_atoms = self.get_Start_End_Atoms()
        #Don't need to calculate direct exchange if there aren't any pairs.
        if len(exchange_atoms) == 0:
            return "Done"
        #Initialize the list of inputs we will parallel process
        task_list = [[''] * 9, []]
        found_keys = {}
        for a_a in exchange_atoms:
            #Get charge of bra atom
            zeff_bra = self.Zeff(a_a)
            #The list of orbitals of bra atom
            orbital_list_bra = self.Orbitals(a_a)
            #The list of bonded atoms to bra atom
            bonded_list = self.get_Bonds(a_a) 
            #Sum over the list of atoms to bra atom
            for a_b in exchange_atoms:
                if a_a != a_b:
                    #Get charge of each ket atom
                    zeff_ket = self.Zeff(a_b)
                    #The list of orbitals of ket atom
                    orbital_list_ket = self.Orbitals(a_b)
                    #Get quantum numbers of bra atom
                    for o_a in orbital_list_bra:
                        na, la, ma_list = self.get_Quantum_Numbers(o_a)
                        #Go through all possible m values of bra atom
                        for ma in ma_list:
                            #Get the quantum numbers of ket atom
                            for o_b in orbital_list_ket:
                                nb, lb, mb_list = self.get_Quantum_Numbers(o_b)
                                #Go through all possible m values of ket atom
                                for mb in mb_list:
                                    #Appends the task list to fit all orbital exchanges
                                    if f"{a_b}({o_b}{mb})-{a_a}({o_a}{ma})" not in found_keys and f"{a_a}({o_a}{ma})-{a_b}({o_b}{mb})" not in found_keys:
                                        found_keys[f"{a_a}({o_a}{ma})-{a_b}({o_b}{mb})"] = 0
                                        task_list[1] += [f"{a_a}({o_a}{ma})-{a_b}({o_b}{mb})"]
                                        next_param = self.calc_de(a_a, f'{o_a}{ma}',a_b, f'{o_b}{mb}',True)
                                        for i in list(range(9)):
                                            task_list[0][i] = task_list[0][i] + next_param[i] + ","
        # Now that we determine all the tasks, we want to use parallel_evaluate to calculate this faster by opening multiple
        # kernals. In theory, this should decrease the processing time by an inverse factor of the number of kernals,
        # so using a supercomputer should make this extremely fast!
        # param = Transpose[{}]
        # Organizes the string to input. Aweful looking but fast enough relative to the actual calculation.
        eval_string = f'{"{{" + task_list[0][0][:-1] + "}"},{"{" + task_list[0][1][:-1] + "}"},{"{" + task_list[0][2][:-1] + "}"},{"{" + task_list[0][3][:-1] + "}"},{"{" + task_list[0][4][:-1] + "}"},{"{" + task_list[0][5][:-1] + "}"},{"{" + task_list[0][6][:-1] + "}"},{"{" + task_list[0][7][:-1] + "}"},{"{" + task_list[0][8][:-1] + "}}"}'
        de_values = self.session.evaluate(f'ParallelMap[Apply[CubicBarbaraDirectExchange], Transpose[{eval_string}]]')
        self.de_values = de_values
        #This shuts down the extra sessions then reloads the session for our future components
        #that don't need to work in parallel.
        self.session.terminate()
        self.session = None
        self.Init_Mathematica()
        for i in list(range(len(de_values))):
            #Checks if it is a complex number and makes it real if it is, as transfer integrals are real objects, so the
            #imaginary component is just rounding error in the system.
            self.found_de[task_list[1][i]] = float(de_values[i])
            #Used to get the atoms
            atoms_ab = task_list[1][i].split("(")
            a_a, a_b = atoms_ab[0], atoms_ab[1].split("-")[-1]
            if f"{a_a}-{a_b}" not in self.DE_ab:
                self.DE_ab[f"{a_a}-{a_b}"] = float(de_values[i])
                self.DE_ab[f"{a_b}-{a_a}"] = float(de_values[i])
            else:
                self.DE_ab[f"{a_a}-{a_b}"] += float(de_values[i])
                self.DE_ab[f"{a_b}-{a_a}"] += float(de_values[i])
                    
    #Calculate all direct exchange integrals with atoms directly bonded to the exchange atom.
    #(Broadly a remenent of a mistake, but keeping around in case it's useful).
    def Calculate_All_Neigbor_de(self):
        self.Init_Mathematica()
        #Initialize the list of inputs we will parallel process
        task_list = [[''] * 9, []]
        found_keys = {}
        #Get a list of starting central atoms
        exchange_atoms = self.get_Start_End_Atoms()
        for a_a in exchange_atoms:
            #Get charge of bra atom
            zeff_bra = self.Zeff(a_a)
            #The list of orbitals of bra atom
            orbital_list_bra = self.Orbitals(a_a)
            #The list of bonded atoms to bra atom
            bonded_list = self.get_Bonds(a_a) 
            #Sum over the list of neighbouring atoms to bra atom
            for a_b in bonded_list:
                #Get charge of each ket atom
                zeff_ket = self.Zeff(a_b)
                #The list of orbitals of ket atom
                orbital_list_ket = self.Orbitals(a_b)
                #Get quantum numbers of bra atom
                for o_a in orbital_list_bra:
                    na, la, ma_list = self.get_Quantum_Numbers(o_a)
                    #Go through all possible m values of bra atom
                    for ma in ma_list:
                        #Get the quantum numbers of ket atom
                        for o_b in orbital_list_ket:
                            nb, lb, mb_list = self.get_Quantum_Numbers(o_b)
                            #Go through all possible m values of ket atom
                            for mb in mb_list:
                                #Appends the task list to fit all orbital exchanges
                                if f"{a_b}({o_b}{mb})-{a_a}({o_a}{ma})" not in found_keys and f"{a_a}({o_a}{ma})-{a_b}({o_b}{mb})" not in found_keys:
                                    found_keys[f"{a_a}({o_a}{ma})-{a_b}({o_b}{mb})"] = 0
                                    task_list[1] += [f"{a_a}({o_a}{ma})-{a_b}({o_b}{mb})"]
                                    next_param = self.calc_de(a_a, f'{o_a}{ma}',a_b, f'{o_b}{mb}',True)
                                    for i in list(range(9)):
                                        task_list[0][i] = task_list[0][i] + next_param[i] + ","
        # Now that we determine all the tasks, we want to use parallel_evaluate to calculate this faster by opening multiple
        # kernals. In theory, this should decrease the processing time by an inverse factor of the number of kernals,
        # so using a supercomputer should make this extremely fast!
        # param = Transpose[{}]
        # Organizes the string to input. Aweful looking but fast enough relative to the actual calculation.
        eval_string = f'{"{{" + task_list[0][0][:-1] + "}"},{"{" + task_list[0][1][:-1] + "}"},{"{" + task_list[0][2][:-1] + "}"},{"{" + task_list[0][3][:-1] + "}"},{"{" + task_list[0][4][:-1] + "}"},{"{" + task_list[0][5][:-1] + "}"},{"{" + task_list[0][6][:-1] + "}"},{"{" + task_list[0][7][:-1] + "}"},{"{" + task_list[0][8][:-1] + "}}"}'
        de_values = self.session.evaluate(f'ParallelMap[Apply[CubicBarbaraDirectExchange], Transpose[{eval_string}]]')
        self.de_Task_List = task_list
        self.de_values = de_values
        #This shuts down the extra sessions then reloads the session for our future components
        #that don't need to work in parallel.
        self.session.terminate()
        self.session = None
        self.Init_Mathematica()
        for i in list(range(len(de_values))):
            #Checks if it is a complex number and makes it real if it is, as transfer integrals are real objects, so the
            #imaginary component is just rounding error in the system.
            self.found_de[task_list[1][i]] = float(de_values[i])
            #Used to get the atoms
            atoms_ab = task_list[1][i].split("(")
            a_a, a_b = atoms_ab[0], atoms_ab[1].split("-")[-1]
            if f"{a_a}-{a_b}" not in self.DE_ab:
                self.DE_ab[f"{a_a}-{a_b}"] = float(de_values[i])
                self.DE_ab[f"{a_b}-{a_a}"] = float(de_values[i])
            else:
                self.DE_ab[f"{a_a}-{a_b}"] += float(de_values[i])
                self.DE_ab[f"{a_b}-{a_a}"] += float(de_values[i])
    
    #Gets the dipole-dipole interaction matrix
    def Dip_Dip_Int(self, atom_a, atom_b):
        x, y, z = self.get_Displacement(atom_a, atom_b)
        r = np.sqrt(x**2 + y**2 + z**2)
        prefactor = ((1/137)**2)/2  #Half Fine Structure Constant Squared
        return np.array([[r**2-3*x**2,-3*x*y,-3*x*z],
                         [-3*y*x,r**2-3*y**2,-3*y*z],
                         [-3*z*x,-3*z*y,r**2-3*z**2]])*(prefactor/r**5)

    #This calculates the Anderson Superexchange for a set of 4 orbitals:
    #(Currently set up just to list the combinations). Note all these combinations need to iturate through all
    #combinations of the m-values for each orbital set, making this an extremely long calculation.
    # Note of Atom Indices: eta,kappa,rho,zeta = a,c,d,b
    def Iso_J(self, a_a, o_a, na, la, ma, a_b, o_b, nb, lb, mb, a_c, o_c, nc, lc, mc, a_d, o_d, nd, ld, md):
        J = 0
        #Gets needed information for getting the requisite information.
        Sa, Sb = [self.Atom_Spin(a_a), self.Atom_Spin(a_b)]
        #This is to ensure that the orbitals at the ends are neither full nor empty.
        if Sa == 0 or Sb == 0:
            return J
        #Initializer of labeling
        del_ab = self.Charge_Transfer(a_a, a_b)
        del_ac = self.Charge_Transfer(a_a, a_c)
        del_ad = self.Charge_Transfer(a_a, a_d)
        #Just makes sure the denominator of J is never 0.
        if 0 in [del_ab, del_ac, del_ad]:
            return J
        else:
            tac = self.t(a_a, f"{o_a}{ma}", a_c, f"{o_c}{mc}")
            tcb = self.t(a_c, f"{o_c}{mc}", a_b, f"{o_b}{mb}")
            tbd = self.t(a_b, f"{o_b}{mb}", a_d, f"{o_d}{md}")
            tda = self.t(a_d, f"{o_d}{md}", a_a, f"{o_a}{ma}")
            #If any transfer integrals are zero, returns zero exchange
            if 0 in [tac,tcb,tbd,tda]:
                J = 0
            else:
                #This gives the anderson component (or anderson interference component if a_c != a_d)
                J = ((tac*tcb*tbd*tda)/(Sa*Sb*del_ab*del_ac*del_ad))
            #This gets the Prett interference components discussed on page 202 of the Barbara text
            #for the symetric isotropic component.
            if a_c != a_d:
                del_bd = self.Charge_Transfer(a_b, a_d)
                del_bc = self.Charge_Transfer(a_b, a_c)
                del_db = self.Charge_Transfer(a_d, a_b)
                del_ca = self.Charge_Transfer(a_c, a_a)
                if 0 in [del_bd, del_bc, del_db, del_ca]:
                    J += 0
                else:
                    tad = self.t(a_a, f"{o_a}{ma}", a_d, f"{o_d}{md}")
                    tbc = self.t(a_b, f"{o_b}{mb}", a_c, f"{o_c}{mc}")
                    tca = self.t(a_c, f"{o_c}{mc}", a_a, f"{o_a}{ma}")
                    tdb = self.t(a_d, f"{o_d}{md}", a_b, f"{o_b}{mb}")
                    J += ((1/np.abs(Sa*Sb*(del_ac*del_bd)))*(((tad*tbc)/del_bc)+((tbc*tad)/del_ad))*(((tdb*tca)/del_ca)+((tca*tdb)/del_db)))
        return J
    
    #This is used to calculate the DM interaction
    # Note of Atom Indices: eta,kappa,rho,zeta = a,c,d,b
    def DM_Vector(self, a_a, o_a, na, la, ma, a_b, o_b, nb, lb, mb, a_c, o_c, nc, lc, mc, a_d, o_d, nd, ld, md):
        #Gets needed information for getting the requisite information.
        DM, DM_ab, DM_ba = [0]*3, [0]*3, [0]*3
        Sa, Sb = [self.Atom_Spin(a_a), self.Atom_Spin(a_b)]
        if Sa == 0 or Sb == 0:
            return DM, DM_ab, DM_ba
        del_ab = self.Charge_Transfer(a_a, a_b)
        del_ca = self.Charge_Transfer(a_c, a_a)
        del_da = self.Charge_Transfer(a_d, a_a)
        del_cb = self.Charge_Transfer(a_c, a_b)
        del_db = self.Charge_Transfer(a_d, a_b)
        #We assume that the differences in orbitals within the atoms are signifigantly smaller then charge transfer
        #energy between two atoms.
        #Just makes sure the denominator of DM is never 0
        #If any transfer integrals are zero, return identically zero vector
        if 0 in [del_ab, del_ca, del_da, del_cb, del_db]:
            return DM, DM_ab, DM_ba
        #Skips if mea degenerate or a lower energy state with respect to mc or if any of the
        #energy differences are 0, indicating a devide by zero error/no transfer (shouldn't happen).
        #Checks for excited states relative to mc for exchange over DM vector.
        coeff = (1/(2*Sa*Sb*del_ab))*((1/(del_ca*del_da))+(1/(del_cb*del_db)))
        SOCC_a = self.SOCC(a_a)
        SOCC_b = self.SOCC(a_b)
        tac = self.t(a_a, f"{o_a}{ma}", a_c, f"{o_c}{mc}")
        tda = self.t(a_d, f"{o_d}{md}", a_a, f"{o_a}{ma}")
        tcb = self.t(a_c, f"{o_c}{mc}", a_b, f"{o_b}{mb}")
        tbd = self.t(a_b, f"{o_b}{mb}", a_d, f"{o_d}{md}")
        if 0 in [tbd,tda,tcb]:
            pass
        else:
            for mea in list(range(-la, la+1)):
                if ma == mea:
                    continue
                del_mamea = self.Delta_12(a_a, o_a, ma, a_a, o_a, mea)
                if del_mamea == 0:
                    continue
                #gets <ma|L|mea> as a list of [lx, ly, lz] components.
                ang_aea = self.Angular_Expect(a_a, na, la, ma, mea)
                if ang_aea == [0]*3:
                    continue
                teac = self.t(a_a, f"{o_a}{mea}", a_c, f"{o_c}{mc}")
                if teac == 0:
                    continue
                #Loops over the x, y, and z terms (use *1j if using Wolfram and *sp.I if Python cubic angular functions)
                for i in list(range(3)):
                    DM_ab[i] = np.real(1j*coeff*SOCC_a*tbd*tda*(ang_aea[i]/del_mamea)*teac*tcb)
                    DM[i] += DM_ab[i]

        if 0 in [tac,tcb,tda]:
            pass
        else:
            for meb in list(range(-lb, lb+1)):
                if mb == meb:
                    continue
                del_mbmeb = self.Delta_12(a_b, o_b, mb, a_b, o_b, meb)
                if del_mbmeb == 0:
                    continue
                #gets <mb|L|meb> as a list of [lx, ly, lz] components.
                ang_beb = self.Angular_Expect(a_b, nb, lb, mb, meb)
                if ang_beb == [0]*3:
                    continue
                tebd = self.t(a_b, f"{o_b}{meb}", a_d, f"{o_d}{md}")
                if tebd == 0:
                    continue
                for i in list(range(3)):
                    DM_ba[i] = np.real(1j*coeff*SOCC_b*tac*tcb*(ang_beb[i]/del_mbmeb)*tebd*tda)
                    DM[i] += -DM_ba[i]
        return DM, DM_ab, DM_ba
                
    #This is used to calculate the symetric anisotropic tensors
    # Note of Atom Indices: eta,kappa,rho,zeta = a,c,d,b
    def AE_Tensor(self, a_a, o_a, na, la, ma, a_b, o_b, nb, lb, mb, a_c, o_c, nc, lc, mc, a_d, o_d, nd, ld, md):
        #Gets needed information for getting the requisite information.
        AE = [0]*9
        Sa, Sb = [self.Atom_Spin(a_a), self.Atom_Spin(a_b)] 
        if Sa == 0 or Sb == 0:
            return AE
        del_ab = self.Charge_Transfer(a_a, a_b)
        del_ac = self.Charge_Transfer(a_a, a_c)
        del_ad = self.Charge_Transfer(a_a, a_d)
        #Just makes sure the denominator of DM is never 0.
        if 0 in [del_ab, del_ac, del_ad]:
            return AE
        else:
            #Skips if mea degenerate or a lower energy state with respect to mc or if any of the
            #energy differences are 0, indicating a devide by zero error/no transfer (shouldn't happen).
            #Checks for excited states relative to mc for exchange over DM vector.
            coeff = (1/(8*Sa*Sb*del_ab*del_ac*del_ad))
            SOCC_a = self.SOCC(a_a)
            SOCC_b = self.SOCC(a_b)
            #These are just set up so we don't have to deal with functions as much. Helps w/ speed a bit
            #and also make it more readable. These are the only 4 that show up.
            tad = self.t(a_a, f"{o_a}{ma}", a_d, f"{o_d}{md}")
            tca = self.t(a_c, f"{o_c}{mc}", a_a, f"{o_a}{ma}")
            tbc = self.t(a_b, f"{o_b}{mb}", a_c, f"{o_c}{mc}")
            tdb = self.t(a_d, f"{o_d}{md}", a_b, f"{o_b}{mb}")
            #This gets the term with mea and meea
            if 0 in [tdb,tbc]:
                pass
            else:
                for mea in list(range(-la, la+1)):
                    if mea == ma:
                        continue
                    del_mamea = self.Delta_12(a_a, o_a, ma, a_a, o_a, mea)
                    #Makes sure it's actually an excited state
                    if del_mamea == 0:
                        continue
                    #gets <ma|L|mea> as a list of [lx, ly, lz] components.
                    #Note: Angular_Expect(a, n, l, m_bra, m_ket) is the formatting of this function.
                    #gets <ma|L|mea> as a list of [lx, ly, lz] components.
                    ang_mamea = self.Angular_Expect(a_a, na, la, ma, mea)
                    if ang_mamea == [0]*3:
                        continue
                    #Cuts down processing resources and makes it more readable.
                    tead = self.t(a_a, f"{o_a}{mea}", a_d, f"{o_d}{md}")
                    if tead == 0:
                        continue
                    #Used to calculate the a-coupling only term.
                    for meea in list(range(-la, la+1)):
                        if meea == ma or meea == mea:
                            continue
                        #Crystal Field Energy
                        del_mameea = self.Delta_12(a_a, o_a, ma, a_a, o_a, meea)
                        if del_mameea == 0:
                            continue
                        #gets <ma|L|mea> as a list of [lx, ly, lz] components.
                        #Note: Angular_Expect(a, n, l, m_bra, m_ket) is the formatting of this function.
                        #gets <meea|L|ma> as a list of [lx, ly, lz] components.
                        ang_meeama = self.Angular_Expect(a_a, na, la, meea, ma)
                        if ang_meeama == [0]*3:
                            continue
                        tceea = self.t(a_c, f"{o_c}{mc}", a_a, f"{o_a}{meea}")
                        if tceea == 0:
                            continue
                        for ang_xyz in range(9):
                            #With i and j, we can get the correct terms for all combinations of L
                            #for the 2 angular terms in the 9 long list at consistent indicies:
                            #[xx, xy, xz, yx, yy, yz, zx, zy, zz]
                            #gets 0(x) for [0,1,2], 1(y) for [3, 4, 5], and 2(z) for [6,7,8]
                            i = math.floor(ang_xyz/3)
                            #gets 0(x) for [0,3,6], 1(y) for [1,4,7], and 2(z) for [2,5,8]
                            j = ang_xyz%3
                            AE[ang_xyz] += np.real(coeff*(SOCC_a**2)*(ang_mamea[i]/del_mamea)*tead*tdb*tbc*tceea*(ang_meeama[j]/del_mameea))
                            
            #This gets the term with meb and meeb
            if 0 in [tad,tca]:
                pass
            else:
                for meb in list(range(-lb, lb+1)):
                    if mb == meb:
                        continue
                    del_mbmeb = self.Delta_12(a_b, o_b, mb, a_b, o_b, meb)
                    if del_mbmeb == 0:
                        continue
                    #Note: Angular_Expect(a, n, l, m_bra, m_ket) is the formatting of this function.
                    #gets <mb|L|meb> as a list of [lx, ly, lz] components.
                    ang_mbmeb = self.Angular_Expect(a_b, nb, lb, mb, meb)
                    if ang_mbmeb == [0]*3:
                        continue
                    #Cuts down processing resources and makes it more readable.
                    tebc = self.t(a_b, f"{o_b}{meb}", a_c, f"{o_c}{mc}")
                    if tebc == 0:
                        continue
                    #Used to calculate the b-coupling only term.
                    for meeb in list(range(-lb, lb+1)):
                        if meeb == mb or meeb == meb:
                            continue
                        del_mbmeeb = self.Delta_12(a_b, o_b, mb, a_b, o_b, meeb)
                        if del_mbmeeb == 0:
                            continue
                        #Note: Angular_Expect(a, n, l, m_bra, m_ket) is the formatting of this function.
                        #gets <meeb|L|mb> as a list of [lx, ly, lz] components.
                        ang_meebmb = self.Angular_Expect(a_b, nb, lb, meeb, mb)
                        if ang_meebmb == [0]*3:
                            continue
                        tdeeb = self.t(a_d, f"{o_d}{md}", a_b, f"{o_b}{meeb}")
                        if tdeeb == 0:
                            continue
                        for ang_xyz in range(9):
                            #With i and j, we can get the correct terms for all combinations of L
                            #for the 2 angular terms in the 9 long list at consistent indicies:
                            #[xx, xy, xz, yx, yy, yz, zx, zy, zz]
                            #gets 0(x) for [0,1,2], 1(y) for [3, 4, 5], and 2(z) for [6,7,8]
                            i = math.floor(ang_xyz/3)
                            #gets 0(x) for [0,3,6], 1(y) for [1,4,7], and 2(z) for [2,5,8]
                            j = ang_xyz%3
                            AE[ang_xyz] += np.real(coeff*(SOCC_b**2)*tad*tdeeb*(ang_meebmb[i]/del_mbmeeb)*(ang_mbmeb[j]/del_mbmeb)*tebc*tca)
                            
            #Now we calculate the other 4 terms, where we only consider mea and meb.
            for mea in range(-la, la+1):
                if ma == mea:
                    continue
                del_mamea = self.Delta_12(a_a, o_a, ma, a_a, o_a, mea)
                if del_mamea == 0:
                    continue
                #Note: Angular_Expect(a, n, l, m_bra, m_ket) is the formatting of this function
                #gets <ma|L|mea> as a list of [lx, ly, lz] components
                ang_mamea = self.Angular_Expect(a_a, na, la, ma, mea)
                ang_meama = self.Angular_Expect(a_a, na, la, mea, ma)
                #If this is all 0, the terms will all be 0.
                if ang_mamea == [0]*3 and ang_meama == [0]*3:
                    continue
                #Cuts down processing resources and makes it more readable.
                tead = self.t(a_a, f"{o_a}{mea}", a_d, f"{o_d}{md}")
                tcea = self.t(a_c, f"{o_c}{mc}", a_a, f"{o_a}{mea}")
                for meb in range(-lb, lb+1):
                    if mb == meb:
                        continue
                    del_mbmeb = self.Delta_12(a_b, o_b, mb, a_b, o_b, meb)
                    if del_mbmeb == 0:
                        continue
                    #Note: Angular_Expect(a, n, l, m_bra, m_ket) is the formatting of this function.
                    #gets <mb|L|meb> as a list of [lx, ly, lz] components.
                    ang_mbmeb = self.Angular_Expect(a_b, nb, lb, mb, meb)
                    ang_mebmb = self.Angular_Expect(a_b, nb, lb, meb, mb)
                    #If this is all 0, the terms will all be 0.
                    if ang_mbmeb == [0]*3 and ang_mebmb == [0]*3:
                        continue
                    #Cuts down processing resources and makes it more readable.
                    tebc = self.t(a_b, f"{o_b}{meb}", a_c, f"{o_c}{mc}")
                    tdeb = self.t(a_d, f"{o_d}{md}", a_b, f"{o_b}{meb}")
                    for ang_xyz in range(9):
                        #With i and j, we can get the correct terms for all combinations of L
                        #for the 2 angular terms in the 9 long list at consistent indicies:
                        #[xx, xy, xz, yx, yy, yz, zx, zy, zz]
                        #gets 0(x) for [0,1,2], 1(y) for [3, 4, 5], and 2(z) for [6,7,8]
                        i = math.floor(ang_xyz/3)
                        #gets 0(x) for [0,3,6], 1(y) for [1,4,7], and 2(z) for [2,5,8]
                        j = ang_xyz%3
                        AE[ang_xyz] += np.real(coeff*SOCC_a*SOCC_b*((ang_mamea[i]/del_mamea)*tead*tdb*(ang_mbmeb[j]/del_mbmeb)*tebc*tca
                                                                    +tad*tdeb*(ang_mebmb[i]/del_mbmeb)*tbc*tcea*(ang_meama[j]/del_mamea)
                                                                    +tad*tdb*(ang_mbmeb[i]/del_mbmeb)*tebc*tcea*(ang_meama[j]/del_mamea)
                                                                    +(ang_mamea[i]/del_mamea)*tead*tdeb*(ang_mebmb[j]/del_mbmeb)*tbc*tca))
        return AE
    
    #This function is used to calculate the symetric isotropic term from Anderson superchange mechanisms
    #as well as the antisymetric anisotropic Dzyaloshinskii–Moriya(DM) vector and symetric anisotropic exchange tensor(AE).
    #Sets up the loops and data storage needed for the Iso_J function, DM_Vector, and AE_Tensor functions.
    #Note that saving the data to this detail is very expensive, so account for data storage taking about 5x
    #the time it takes to calculate the transfer integrals.
    #If l_detail = True, then we account for the lalcldlb and lalb data sets.
    #If m_detail = True, then we also account for all the mamcmdmb, malcldmb, and lamcmdlb data sets.
    def Superexchange_Terms(self, l_detail = False, m_detail = False):
        if len(self.structured_pairs) == 0:
            print("Calculating Transfer Integrals:")
            start = time.perf_counter()
            self.Calculate_All_t()
            print(f"Transfer Integrals Complete: {round((time.perf_counter() - start)/60, 1)} min")
            print("Calculating Exchange Terms")
        start = time.perf_counter()
        #Uses the preorganized set of 
        for structured_pair in self.structured_pairs:
            atom_a, na, la, ma = structured_pair[0]
            orbital_a = f'{na}{"spdfghij"[la]}'
            atom_b, nb, lb, mb = structured_pair[1]
            orbital_b = f'{nb}{"spdfghij"[lb]}'
            #Gets needed information for getting the requisite information.
            Sa, Sb = [self.Atom_Spin(atom_a), self.Atom_Spin(atom_b)]
            #This is to ensure that the orbitals at the ends are neither full nor empty. If one is, we skip,
            #as without doing this, we devide by 0 in all components.
            if Sa == 0 or Sb == 0:
                continue
            atom_c, nc, lc, mc = structured_pair[2]
            orbital_c = f'{nc}{"spdfghij"[lc]}'
            atom_d, nd, ld, md = structured_pair[3]
            orbital_d = f'{nd}{"spdfghij"[ld]}'
            #Simply used to condence if statements for faster processing times and denser code when assigning keys.
            is_acadocod = ((atom_c != atom_d) or (orbital_c != orbital_d))
            is_acadocodmcmd = ((is_acadocod) or (mc != md))
            #Gets the symetric isotropic J.
            J = self.Iso_J(atom_a, orbital_a, na, la, ma, atom_b, orbital_b, nb, lb, mb, 
                           atom_c, orbital_c, nc, lc, mc, atom_d, orbital_d, nd, ld, md)
            if J != 0:
                #Now saves the data.
                if m_detail == True:
                    self.J_mamcmdmb[f"{atom_a}({orbital_a}{ma})-{atom_c}({orbital_c}{mc})" + f"-{atom_d}({orbital_d}{md})"*is_acadocodmcmd + f"-{atom_b}({orbital_b}{mb})"] = J
                    self.J_malcldmb[f"{atom_a}({orbital_a}{ma})-{atom_c}({orbital_c})" + f"-{atom_d}({orbital_d})"*is_acadocod + f"-{atom_b}({orbital_b}{mb})"] += J
                    self.J_lamcmdlb[f"{atom_a}({orbital_a})-{atom_c}({orbital_c}{mc})" + f"-{atom_d}({orbital_d}{md})"*is_acadocodmcmd + f"-{atom_b}({orbital_b})"] += J
                if l_detail == True:
                    self.J_lalcldlb[f"{atom_a}({orbital_a})-{atom_c}({orbital_c})" + f"-{atom_d}({orbital_d})"*is_acadocod + f"-{atom_b}({orbital_b})"] += J
                    self.J_lalb[f"{atom_a}({orbital_a})-{atom_b}({orbital_b})"] += J
                self.J_ab[f"{atom_a}-{atom_b}"] += J

            AE = self.AE_Tensor(atom_a, orbital_a, na, la, ma, atom_b, orbital_b, nb, lb, mb, 
                                atom_c, orbital_c, nc, lc, mc, atom_d, orbital_d, nd, ld, md)
            if AE != [0]*9:
                #Just uses maps to add the data slightly faster.
                if m_detail == True:
                    self.AE_mamcmdmb[f"{atom_a}({orbital_a}{ma})-{atom_c}({orbital_c}{mc})" + f"-{atom_d}({orbital_d}{md})"*is_acadocodmcmd + f"-{atom_b}({orbital_b}{mb})"] = AE
                    self.AE_malcldmb[f"{atom_a}({orbital_a}{ma})-{atom_c}({orbital_c})" + f"-{atom_d}({orbital_d})"*is_acadocod + f"-{atom_b}({orbital_b}{mb})"] = list(map(self.Add_List_Val, self.AE_malcldmb[f"{atom_a}({orbital_a}{ma})-{atom_c}({orbital_c})" + f"-{atom_d}({orbital_d})"*is_acadocod + f"-{atom_b}({orbital_b}{mb})"], AE))
                    self.AE_lamcmdlb[f"{atom_a}({orbital_a})-{atom_c}({orbital_c}{mc})" + f"-{atom_d}({orbital_d}{md})"*is_acadocodmcmd + f"-{atom_b}({orbital_b})"] = list(map(self.Add_List_Val, self.AE_lamcmdlb[f"{atom_a}({orbital_a})-{atom_c}({orbital_c}{mc})" + f"-{atom_d}({orbital_d}{md})"*is_acadocodmcmd + f"-{atom_b}({orbital_b})"], AE))
                if l_detail == True:
                    self.AE_lalcldlb[f"{atom_a}({orbital_a})-{atom_c}({orbital_c})" + f"-{atom_d}({orbital_d})"*is_acadocod + f"-{atom_b}({orbital_b})"] = list(map(self.Add_List_Val, self.AE_lalcldlb[f"{atom_a}({orbital_a})-{atom_c}({orbital_c})" + f"-{atom_d}({orbital_d})"*is_acadocod + f"-{atom_b}({orbital_b})"], AE))
                    self.AE_lalb[f"{atom_a}({orbital_a})-{atom_b}({orbital_b})"] = list(map(self.Add_List_Val, self.AE_lalb[f"{atom_a}({orbital_a})-{atom_b}({orbital_b})"], AE))
                self.AE_ab[f"{atom_a}-{atom_b}"] = list(map(self.Add_List_Val, self.AE_ab[f"{atom_a}-{atom_b}"], AE))
            #Checks that cation is the same for c and d or else don't run the DM vector
            #Also needs 2 different atoms, since Sa x Sb = 0 when Sa = Sb for the hamiltonian term about d_ab.
            #calculation as it isn't applicable.
            if atom_c == atom_d and atom_a != atom_b:
                DM, DM_ab, DM_ba = self.DM_Vector(atom_a, orbital_a, na, la, ma, atom_b, orbital_b, nb, lb, mb, 
                                                  atom_c, orbital_c, nc, lc, mc, atom_d, orbital_d, nd, ld, md)
                if DM != [0]*3:
                    #Saves to database.
                    if m_detail == True:
                        self.DM_mamcmdmb_tot[f"{atom_a}({orbital_a}{ma})-{atom_c}({orbital_c}{mc})" + f"-{atom_c}({orbital_d}{md})"*is_acadocodmcmd + f"-{atom_b}({orbital_b}{mb})"] = DM
                        self.DM_malcldmb_tot[f"{atom_a}({orbital_a}{ma})-{atom_c}({orbital_c})" + f"-{atom_c}({orbital_d})"*is_acadocod + f"-{atom_b}({orbital_b}{mb})"] = list(map(self.Add_List_Val, self.DM_malcldmb_tot[f"{atom_a}({orbital_a}{ma})-{atom_c}({orbital_c})" + f"-{atom_c}({orbital_d})"*is_acadocod + f"-{atom_b}({orbital_b}{mb})"], DM))
                        self.DM_lamcmdlb_tot[f"{atom_a}({orbital_a})-{atom_c}({orbital_c}{mc})" + f"-{atom_c}({orbital_d}{md})"*is_acadocodmcmd + f"-{atom_b}({orbital_b})"] = list(map(self.Add_List_Val, self.DM_lamcmdlb_tot[f"{atom_a}({orbital_a})-{atom_c}({orbital_c}{mc})" + f"-{atom_c}({orbital_d}{md})"*is_acadocodmcmd + f"-{atom_b}({orbital_b})"], DM))
                    if l_detail == True:
                        self.DM_lalcldlb_tot[f"{atom_a}({orbital_a})-{atom_c}({orbital_c})" + f"-{atom_c}({orbital_d})"*is_acadocod + f"-{atom_b}({orbital_b})"] = list(map(self.Add_List_Val, self.DM_lalcldlb_tot[f"{atom_a}({orbital_a})-{atom_c}({orbital_c})" + f"-{atom_c}({orbital_d})"*is_acadocod + f"-{atom_b}({orbital_b})"], DM))
                        self.DM_lalb_tot[f"{atom_a}({orbital_a})-{atom_b}({orbital_b})"] = list(map(self.Add_List_Val, self.DM_lalb_tot[f"{atom_a}({orbital_a})-{atom_b}({orbital_b})"], DM))
                    self.DM_ab_tot[f"{atom_a}-{atom_b}"] = list(map(self.Add_List_Val, self.DM_ab_tot[f"{atom_a}-{atom_b}"], DM))
                if DM_ab != [0]*3:
                    #Dictionaries for splitting and capturing the antisymetric direction's values.
                    if m_detail == True:
                        self.DM_mamcmdmb[f"{atom_a}({orbital_a}{ma})-{atom_c}({orbital_c}{mc})" + f"-{atom_c}({orbital_d}{md})"*is_acadocodmcmd + f"-{atom_b}({orbital_b}{mb})"] = DM_ab
                        self.DM_malcldmb[f"{atom_a}({orbital_a}{ma})-{atom_c}({orbital_c})" + f"-{atom_c}({orbital_d})"*is_acadocod + f"-{atom_b}({orbital_b}{mb})"] = list(map(self.Add_List_Val, self.DM_malcldmb[f"{atom_a}({orbital_a}{ma})-{atom_c}({orbital_c})" + f"-{atom_c}({orbital_d})"*is_acadocod + f"-{atom_b}({orbital_b}{mb})"], DM_ab))
                        self.DM_lamcmdlb[f"{atom_a}({orbital_a})-{atom_c}({orbital_c}{mc})" + f"-{atom_c}({orbital_d}{md})"*is_acadocodmcmd + f"-{atom_b}({orbital_b})"] = list(map(self.Add_List_Val, self.DM_lamcmdlb[f"{atom_a}({orbital_a})-{atom_c}({orbital_c}{mc})" + f"-{atom_c}({orbital_d}{md})"*is_acadocodmcmd + f"-{atom_b}({orbital_b})"], DM_ab))
                    if l_detail == True:
                        self.DM_lalcldlb[f"{atom_a}({orbital_a})-{atom_c}({orbital_c})" + f"-{atom_c}({orbital_d})"*is_acadocod + f"-{atom_b}({orbital_b})"] = list(map(self.Add_List_Val, self.DM_lalcldlb[f"{atom_a}({orbital_a})-{atom_c}({orbital_c})" + f"-{atom_c}({orbital_d})"*is_acadocod + f"-{atom_b}({orbital_b})"], DM_ab))
                        self.DM_lalb[f"{atom_a}({orbital_a})-{atom_b}({orbital_b})"] = list(map(self.Add_List_Val, self.DM_lalb[f"{atom_a}({orbital_a})-{atom_b}({orbital_b})"], DM_ab))
                    self.DM_ab[f"{atom_a}-{atom_b}"] = list(map(self.Add_List_Val, self.DM_ab[f"{atom_a}-{atom_b}"], DM_ab))
                if DM_ba != [0]*3:
                    #Dictionaries for splitting and capturing the antisymetric direction's values.
                    if m_detail == True:
                        self.DM_mbmcmdma[f"{atom_a}({orbital_a}{ma})-{atom_c}({orbital_c}{mc})" + f"-{atom_c}({orbital_d}{md})"*is_acadocodmcmd + f"-{atom_b}({orbital_b}{mb})"] = DM_ba
                        self.DM_mblcldma[f"{atom_a}({orbital_a}{ma})-{atom_c}({orbital_c})" + f"-{atom_c}({orbital_d})"*is_acadocod + f"-{atom_b}({orbital_b}{mb})"] = list(map(self.Add_List_Val, self.DM_mblcldma[f"{atom_a}({orbital_a}{ma})-{atom_c}({orbital_c})" + f"-{atom_c}({orbital_d})"*is_acadocod + f"-{atom_b}({orbital_b}{mb})"], DM_ba))
                        self.DM_lbmcmdla[f"{atom_a}({orbital_a})-{atom_c}({orbital_c}{mc})" + f"-{atom_c}({orbital_d}{md})"*is_acadocodmcmd + f"-{atom_b}({orbital_b})"] = list(map(self.Add_List_Val, self.DM_lbmcmdla[f"{atom_a}({orbital_a})-{atom_c}({orbital_c}{mc})" + f"-{atom_c}({orbital_d}{md})"*is_acadocodmcmd + f"-{atom_b}({orbital_b})"], DM_ba))
                    if l_detail == True:
                        self.DM_lblcldla[f"{atom_a}({orbital_a})-{atom_c}({orbital_c})" + f"-{atom_c}({orbital_d})"*is_acadocod + f"-{atom_b}({orbital_b})"] = list(map(self.Add_List_Val, self.DM_lblcldla[f"{atom_a}({orbital_a})-{atom_c}({orbital_c})" + f"-{atom_c}({orbital_d})"*is_acadocod + f"-{atom_b}({orbital_b})"], DM_ba))
                        self.DM_lbla[f"{atom_a}({orbital_a})-{atom_b}({orbital_b})"] = list(map(self.Add_List_Val, self.DM_lbla[f"{atom_a}({orbital_a})-{atom_b}({orbital_b})"], DM_ba))
                    self.DM_ba[f"{atom_a}-{atom_b}"] = list(map(self.Add_List_Val, self.DM_ba[f"{atom_a}-{atom_b}"], DM_ba))
        #Used to notify when the calculation is finished. Then returns how long the calculation took in minutes.
        print(f"Exchange Terms Complete: {round((time.perf_counter() - start)/60, 2)} min")
        self.session.terminate()
        self.session = None

    ##########################################
    #Definitions for making files of current data
    #If m_detail = True, then we also account for all the mamcmdmb, malcldmb, and lamcmdlb data sets.
    def Export_Data(self, l_detail = False, m_detail = False):
        #Makes a file in the same directory as the python code for storing data files.
        current_directory = os.getcwd()
        file_title = f'{self.data_sheet_name} data {datetime.datetime.now().strftime("%I %M%p on %B %d %Y")}'
        filepath = os.path.join(current_directory, f'{self.data_sheet_name} data {file_title}')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        with open(os.path.join(filepath, 'transfer integrals.csv'), 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            #Column headers for indexing and readability.
            wr.writerow(["Label", "Transfer Integral"])
            #This is used
            for word in self.found_t:
                wr.writerow([word, self.found_t[word]])
        if len(self.DE_ab) != 0:    
            with open(os.path.join(filepath, 'DE_ab.csv'), 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(["Label", "Direct Exchange Value"])
                for word in self.DE_ab:
                    wr.writerow([word, self.DE_ab[word]])
                
        with open(os.path.join(filepath, 'J_ab.csv'), 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(["Label", "Anderson Exchange Value"])
            for word in self.J_ab:
                wr.writerow([word, self.J_ab[word]])
                
        if l_detail == True:
            with open(os.path.join(filepath, 'J_lalb.csv'), 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(["Label", "Anderson Exchange Value"])
                for word in self.J_lalb:
                    wr.writerow([word, self.J_lalb[word]])

            with open(os.path.join(filepath, 'J_lalcldlb.csv'), 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(["Label", "Anderson Exchange Value"])
                for word in self.J_lalcldlb:
                    wr.writerow([word, self.J_lalcldlb[word]])
        if m_detail == True:        
            with open(os.path.join(filepath, 'J_lamcmdlb.csv'), 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(["Label", "Anderson Exchange Value"])
                for word in self.J_lamcmdlb:
                    wr.writerow([word, self.J_lamcmdlb[word]])

            with open(os.path.join(filepath, 'J_malcldmb.csv'), 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(["Label", "Anderson Exchange Value"])
                for word in self.J_malcldmb:
                    wr.writerow([word, self.J_malcldmb[word]])

            with open(os.path.join(filepath, 'J_mamcmdmb.csv'), 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(["Label", "Anderson Exchange Value"])
                for word in self.J_mamcmdmb:
                    wr.writerow([word, self.J_mamcmdmb[word]])
                
        with open(os.path.join(filepath, 'DM_ab.csv'), 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(["Label: Coordinates", "Total DM Vector", "=","DM Vector a->b", "-","DM Vector b->a"])
            for word in self.DM_ab_tot:
                wr.writerow([word + ": x", self.DM_ab_tot[word][0], "=",self.DM_ab[word][0], "-", self.DM_ba[word][0]])
                wr.writerow([word + ": y", self.DM_ab_tot[word][1], "=",self.DM_ab[word][1], "-", self.DM_ba[word][1]])
                wr.writerow([word + ": z", self.DM_ab_tot[word][2], "=",self.DM_ab[word][2], "-", self.DM_ba[word][2]])
        
        if l_detail == True:
            with open(os.path.join(filepath, 'DM_lalb.csv'), 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(["Label: Coordinates", "Total DM Vector", "=","DM Vector a->b", "-","DM Vector b->a"])
                for word in self.DM_lalb:
                    wr.writerow([word + ": x", self.DM_lalb_tot[word][0], "=",self.DM_lalb[word][0], "-", self.DM_lbla[word][0]])
                    wr.writerow([word + ": y", self.DM_lalb_tot[word][1], "=",self.DM_lalb[word][1], "-", self.DM_lbla[word][1]])
                    wr.writerow([word + ": z", self.DM_lalb_tot[word][2], "=",self.DM_lalb[word][2], "-", self.DM_lbla[word][2]])

            with open(os.path.join(filepath, 'DM_lalcldlb.csv'), 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(["Label: Coordinates", "Total DM Vector", "=","DM Vector a->b", "-","DM Vector b->a"])
                for word in self.DM_lalcldlb:
                    wr.writerow([word + ": x", self.DM_lalcldlb_tot[word][0], "=",self.DM_lalcldlb[word][0], "-", self.DM_lblcldla[word][0]])
                    wr.writerow([word + ": y", self.DM_lalcldlb_tot[word][1], "=",self.DM_lalcldlb[word][1], "-", self.DM_lblcldla[word][1]])
                    wr.writerow([word + ": z", self.DM_lalcldlb_tot[word][2], "=",self.DM_lalcldlb[word][2], "-", self.DM_lblcldla[word][2]])

        if m_detail == True:
            with open(os.path.join(filepath, 'DM_lamcmdlb.csv'), 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(["Label: Coordinates", "Total DM Vector", "=","DM Vector a->b", "-","DM Vector b->a"])
                for word in self.DM_lamcmdlb:
                    wr.writerow([word + ": x", self.DM_lamcmdlb_tot[word][0], "=",self.DM_lamcmdlb[word][0], "-", self.DM_lbmcmdla[word][0]])
                    wr.writerow([word + ": y", self.DM_lamcmdlb_tot[word][1], "=",self.DM_lamcmdlb[word][1], "-", self.DM_lbmcmdla[word][1]])
                    wr.writerow([word + ": z", self.DM_lamcmdlb_tot[word][2], "=",self.DM_lamcmdlb[word][2], "-", self.DM_lbmcmdla[word][2]])

            with open(os.path.join(filepath, 'DM_malcldmb.csv'), 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(["Label: Coordinates", "Total DM Vector", "=","DM Vector a->b", "-","DM Vector b->a"])
                for word in self.DM_malcldmb:
                    wr.writerow([word + ": x", self.DM_malcldmb_tot[word][0], "=",self.DM_malcldmb[word][0], "-", self.DM_mblcldma[word][0]])
                    wr.writerow([word + ": y", self.DM_malcldmb_tot[word][1], "=",self.DM_malcldmb[word][1], "-", self.DM_mblcldma[word][1]])
                    wr.writerow([word + ": z", self.DM_malcldmb_tot[word][2], "=",self.DM_malcldmb[word][2], "-", self.DM_mblcldma[word][2]])

            with open(os.path.join(filepath, 'DM_mamcmdmb.csv'), 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(["Label: Coordinates", "Total DM Vector", "=","DM Vector a->b", "-","DM Vector b->a"])
                for word in self.DM_mamcmdmb:
                    wr.writerow([word + ": x", self.DM_mamcmdmb_tot[word][0], "=",self.DM_mamcmdmb[word][0], "-", self.DM_mbmcmdma[word][0]])
                    wr.writerow([word + ": y", self.DM_mamcmdmb_tot[word][1], "=",self.DM_mamcmdmb[word][1], "-", self.DM_mbmcmdma[word][1]])
                    wr.writerow([word + ": z", self.DM_mamcmdmb_tot[word][2], "=",self.DM_mamcmdmb[word][2], "-", self.DM_mbmcmdma[word][2]])

        AE_col_labels = ["Label", "xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]
        with open(os.path.join(filepath, 'AE_ab.csv'), 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(AE_col_labels)
            for word in self.AE_ab:
                cur_AE_tensor = self.AE_ab[word]
                wr.writerow([word] + cur_AE_tensor[0:3] + cur_AE_tensor[3:6] + cur_AE_tensor[6:9])
        if l_detail == True:        
            with open(os.path.join(filepath, 'AE_lalb.csv'), 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(AE_col_labels)
                for word in self.AE_lalb:
                    cur_AE_tensor = self.AE_lalb[word]
                    wr.writerow([word] + cur_AE_tensor[0:3] + cur_AE_tensor[3:6] + cur_AE_tensor[6:9])

            with open(os.path.join(filepath, 'AE_lalcldlb.csv'), 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(AE_col_labels)
                for word in self.AE_lalcldlb:
                    cur_AE_tensor = self.AE_lalcldlb[word]
                    wr.writerow([word] + cur_AE_tensor[0:3] + cur_AE_tensor[3:6] + cur_AE_tensor[6:9])
        if m_detail == True:        
            with open(os.path.join(filepath, 'AE_lamcmdlb.csv'), 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(AE_col_labels)
                for word in self.AE_lamcmdlb:
                    cur_AE_tensor = self.AE_lamcmdlb[word]
                    wr.writerow([word] + cur_AE_tensor[0:3] + cur_AE_tensor[3:6] + cur_AE_tensor[6:9])

            with open(os.path.join(filepath, 'AE_malcldmb.csv'), 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(AE_col_labels)
                for word in self.AE_malcldmb:
                    cur_AE_tensor = self.AE_malcldmb[word]
                    wr.writerow([word] + cur_AE_tensor[0:3] + cur_AE_tensor[3:6] + cur_AE_tensor[6:9])

            with open(os.path.join(filepath, 'AE_mamcmdmb.csv'), 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(AE_col_labels)
                for word in self.AE_mamcmdmb:
                    cur_AE_tensor = self.AE_mamcmdmb[word]
                    wr.writerow([word] + cur_AE_tensor[0:3] + cur_AE_tensor[3:6] + cur_AE_tensor[6:9])
        
    #################################
    #Tools for Analysis
    #################################
    #Gets the coordinate value of an atom in angstrums for X, Y, or Z.
    def get_Atom_Coord(self, atom, axis):
        if axis in ["X", "Y", "Z"]:
            return self.molecule.loc[self.Atom_Index(atom), f"{axis}"]
        else:
            raise Exception(f"In get_Atom_Coord, axis must be either X, Y, or Z. {axis} was input.")
    #This is used to change the position of an atom by some shift along a specific axis.
    #Note, to reset your molecule to the base coords, you can jsut reload the excel sheet
    #using self.molecule = self.get_Molecule_Data(self.data_sheet_name)
    #Note that X, Y, and Z are in angstrums.
    def Shift_Atom_Coord(self, atom, axis, shift):
        if axis in ["X", "Y", "Z"]:
            self.molecule.loc[self.Atom_Index(atom), f"{axis}"] += shift
        else:
            raise Exception(f"In Shift_Atom_Coord, axis must be either X, Y, or Z. {axis} was input.")
    
    #This sets the coordinate of an axis to a specific value.L
    def Set_Atom_Coord(self, atom, axis, coords):
        if axis in ["X", "Y", "Z"]:
            self.molecule.loc[self.Atom_Index(atom), f"{axis}"] = coords
        else:
            raise Exception(f"In Shift_Atom_Coord, axis must be either X, Y, or Z. {axis} was input.")
    
    #This sets the x, y, and z coordinates of an axis to specific values. Input a float list [x, y, z].
    #Note that the coordinates should be in units of angstrums and are converted to meters by the system
    #for ease of input later.
    def Set_Atom_XYZ(self, atom, xyz):
        #Float is applied to all to make sure that the type is valid
        self.molecule.loc[self.Atom_Index(atom), "X"] = float(xyz[0])
        self.molecule.loc[self.Atom_Index(atom), "Y"] = float(xyz[1])
        self.molecule.loc[self.Atom_Index(atom), "Z"] = float(xyz[2])
        
    #Returns the frobenus norm of a matrix
    def Frob_Norm(self, matrix):
        return np.linalg.norm(matrix,ord='fro')
    
    #Get the total exchange matrix between 2 exchange atoms
    def get_Total_Exchange_Matrix(self, atom_a, atom_b):
        #These give the direct exchange and anderson superexchange components.
        #J_dir = (self.DE_ab[f"{atom_a}-{atom_b}"]/3)*np.array([[1,0,0],[0,1,0],[0,0,1]])
        J_iso = (self.J_ab[f"{atom_a}-{atom_b}"]/3)*np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.J_Matrix = J_iso
        #Calculates the dipole-dipole interaction
        D_dip = self.Dip_Dip_Int(atom_a, atom_b)
        self.D_dip = D_dip
        #Gets the Symmetric anisotropic exchange matrix and reorganizes it into a np.array form for use.
        AE_list = self.AE_ab[f"{atom_a}-{atom_b}"]
        D_ae = np.array([[AE_list[0],AE_list[1],AE_list[2]],[AE_list[3],AE_list[4],AE_list[5]],[AE_list[6],AE_list[7],AE_list[8]]])
        self.Dae_Matrix = D_ae
        #Gets the antisymmetric anisotropic exchange component.
        vec_d = self.DM_ab_tot[f"{atom_a}-{atom_b}"]
        Dm = np.array([[0,vec_d[2],-vec_d[1]],[-vec_d[2],0,vec_d[0]],[vec_d[1],-vec_d[0],0]])
        self.Dm_Matrix = Dm
        #Total Exchange Matrix
        return J_iso+D_ae+Dm
        
    #Sets the first atom to the origin and aligns the 2nd atom with the positive x axis. Then shifts all
    #other atoms in the molecule such that they maintain the original position relative to the first atom
    #This is useful when optimizing the positioning with respect to 2 atoms, such as in Optimize_Pos.
    def Align_Atoms(self, a_center, a_x_axis):
        a_c = self.get_Coords(a_center)
        #Shifts all atoms such that atom center is centered at the origin.
        atom_list = self.molecule["Atom"]
        for atom in self.molecule["Atom"]:
            self.Shift_Atom_Coord(atom, "X", -a_c[0])
            self.Shift_Atom_Coord(atom, "Y", -a_c[1])
            self.Shift_Atom_Coord(atom, "Z", -a_c[2])
        #Next, we want to rotate all atoms such that a_x_axis aligns with the x-axis.
        #We do this by finding the angles from x and z axis in the xz and xy planes, respecitvely.
        a_x = self.get_Coords(a_x_axis)
        if a_x[0] == 0:
            theta_rz = np.pi/2
        else:
            theta_rz = np.arctan(a_x[1]/a_x[0])
        if a_x[0]*np.cos(theta_rz) + a_x[1]*np.sin(theta_rz) == 0:
            theta_ry = np.pi/2
        else:
            theta_ry = np.arctan(-a_x[2]/(a_x[0]*np.cos(theta_rz) + a_x[1]*np.sin(theta_rz)))
        #Rotation matrix about the y-axis. The angle to the x-axis when projected onto the xz-plane
        #allows us to rotate our vector into the x axis along the xz-plane
        R_y = np.array([[np.cos(theta_ry),0,-np.sin(theta_ry)],[0,1,0],[np.sin(theta_ry),0,np.cos(theta_ry)]])
        #Rotation matrix about the z-axis. The angle to the y-axis when projected onto the xy-plane
        #allows us to rotate our vector into the y axis along the xy-plane and then rotate clockwise a further
        #90 degrees to align with the x-axis. 
        R_z = np.array([[np.cos(theta_rz),np.sin(theta_rz),0],[np.sin(theta_rz),-np.cos(theta_rz),0],[0,0,1]])
        #Uses the rotation matricies to rotate all the atoms equally about the origin (so atom_center).
        #This is such that a_x_axis is rotated in line with the positive x-axis.
        for atom in self.molecule["Atom"]:
            self.Set_Atom_XYZ(atom, [round(coord, 3) for coord in (list(R_y@R_z@np.array(self.get_Coords(atom))))])

# Functions outside of the class
def pylist_to_mathematica(resultlist):
    'Changes list from python to mathematica format for points (x,y,z,f(x,y,z)'
    # Round results to 3 sig figs
    for sublist in resultlist:
        sublist[-1] = f'{sublist[-1]:.2e}'
    # Change string from python list format, to Mathematica
    resultstring = str(resultlist)
    resultstringclean = resultstring.replace("'","").replace(" ","").replace("[","{").replace("]","}").replace("e","*^")
    return resultstringclean
