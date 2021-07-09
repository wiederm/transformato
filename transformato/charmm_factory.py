import datetime
from os import stat
from transformato.constants import temperature
from simtk import unit


class CharmmFactory:
    """Class to build the string needed to create a CHARMM input and streaming file"""

    def __init__(self, configuration: dict, structure: str) -> None:

        self.configuration = configuration
        self.structure = structure
        self.vdw_switching_keyword = self._check_switching_function()

    def _get_simulations_parameters(self):
        prms = {}
        for key in self.configuration["simulation"]["parameters"]:
            prms[key] = self.configuration["simulation"]["parameters"][key]
        return prms

    def _check_switching_function(self) -> str:
        prms = self._get_simulations_parameters()
        vdw = prms.get("vdw", "Force-switch")  # default is vfswitch
        if vdw.lower() == "force-switch":
            return "vfswitch"
        elif vdw.lower() == "switch":
            return "vswitch"
        elif vdw.lower() == "no-switch":  # not implemented
            return ""
            raise NotImplementedError()

    def generate_CHARMM_postprocessing_files(self, env: str) -> str:

        charmm_postprocessing_script = self._get_CHARMM_postprocessing_header(env)
        if env == "vacuum":
            charmm_postprocessing_script += (
                self._get_CHARMM_vacuum_postprocessing_body()
            )
        elif env == "waterbox":
            charmm_postprocessing_script += (
                self._get_CHARMM_waterbox_postprocessing_body()
            )
        else:
            raise NotImplementedError(f"Something went wrong with {env}.")

        return charmm_postprocessing_script

    def generate_CHARMM_production_files(self, env: str) -> str:
        """Body of the CHARMM file with option for gas pahse, waterbox with vswitch and vfswitch"""

        charmm_production_script = self._get_CHARMM_production_header(env)
        if env == "vacuum":
            charmm_production_script += self._get_CHARMM_vacuum_production_body()
        elif env == "waterbox":
            charmm_production_script += self._get_CHARMM_waterbox_production_body()
        else:
            raise NotImplementedError(f"Something went wrong with {env}.")

        return charmm_production_script

    @staticmethod
    def build_reduced_toppar(tlc: str) -> str:
        date = datetime.date.today()
        toppar = f"""* Simplified toppar script 
* Version from {date} 
*

! Read Protein Topology and Parameter 
open read card unit 10 name ./toppar/top_all36_prot.rtf 
read  rtf card unit 10 
    
open read card unit 20 name ./toppar/par_all36m_prot.prm 
read para card unit 20 flex 

! Read Nucleic Acids 
open read card unit 10 name ./toppar/top_all36_na.rtf 
read  rtf card unit 10 append 
    
open read card unit 20 name ./toppar/par_all36_na.prm 
read para card unit 20 append flex
    
! Read Carbohydrates 
open read card unit 10 name ./toppar/top_all36_carb.rtf 
read  rtf card unit 10 append 
    
open read card unit 20 name ./toppar/par_all36_carb.prm 
read para card unit 20 append flex 

! Read Lipids 
open read card unit 10 name ./toppar/top_all36_lipid.rtf 
read  rtf card unit 10 append 
    
open read card unit 20 name ./toppar/par_all36_lipid.prm 
read para card unit 20 append flex
    
!Read CGENFF 
open read card unit 10 name ./toppar/top_all36_cgenff.rtf 
read  rtf card unit 10 append 
    
open read card unit 20 name ./toppar/par_all36_cgenff.prm 
read para card unit 20 append flex
    
! Additional topologies and parameters for water and ions 
stream ./toppar/toppar_water_ions.str

! Read {tlc} RTF 
open read unit 10 card name {tlc}_g.rtf 
read rtf card unit 10 append

! Read {tlc} prm 
open read unit 10 card name {tlc}.prm 
read para card unit 10 append flex

! Read dummy_atom RTF 
open read unit 10 card name dummy_atom_definitions.rtf 
read rtf card unit 10 append

! Read dummy_atom prm 
open read unit 10 card name dummy_parameters.prm 
read para card unit 10 append flex

"""
        return toppar

    def _get_CHARMM_production_header(self, env: str) -> str:
        intermediate_filename = self.configuration["system"][self.structure][env][
            "intermediate-filename"
        ]

        header = f"""*Version September 2020 
*Run script for CHARMM jobs from transformato 
*

! Read topology and parameter files 
stream charmm_toppar.str 

! Read PSF 
open read unit 10 card name {intermediate_filename}.psf 
read psf  unit 10 card

! Read Coordinate 
open read unit 10 card name {intermediate_filename}.crd 
read coor unit 10 card
        """
        return header

    def _get_CHARMM_vacuum_postprocessing_body(self) -> str:
        body = f"""
coor orie sele all end ! put the molecule at the origin

set ctofnb 990.
set ctonnb 980.
set cutnb  1000.

nbonds ctonnb @ctonnb ctofnb @ctofnb cutnb @cutnb -
  atom swit vatom vswitch -
  inbfrq 1 

energy

energy inbfrq 0

scalar fbeta set 5. sele all end

open read file unit 41 name ../traj.dcd
traj query unit 41

set start ?start
set nframes ?nfile
set skip ?skip

set nframes @nframes !?nfile
traj firstu 41 nunit 1 begi @start skip @skip stop @nframes

open form writ unit 51 name ener_vac.log
echu 51
set idx 0
label loop
traj read
energy
echo ?ener
incr idx by 1
if @idx .lt. @nframes goto loop
   
  
stop"""

        return body

    def _get_CHARMM_vacuum_production_body(self) -> str:
        ##### gas phase ######
        nstep = self.configuration["simulation"]["parameters"]["nstep"]
        nstout = self.configuration["simulation"]["parameters"]["nstout"]
        nstdcd = self.configuration["simulation"]["parameters"]["nstdcd"]
        switch = self.vdw_switching_keyword

        body = f"""
!coor orie sele all end ! put the molecule at the origin

MMFP
GEO rcm sphere -
    Xref 0.0 Yref 0.0 Zref 0.0 XDIR 1.0 YDIR 1.0 ZDIR 1.0 -
    harmonic FORCE 1.0 select .not. ( hydrogen .or. resname TIP3 ) end
END

set ctofnb 990.
set ctonnb 980.
set cutnb  1000.

nbonds ctonnb @ctonnb ctofnb @ctofnb cutnb @cutnb -
  atom swit vatom {switch} -
  inbfrq 1 

energy   inbfrq 1
energy   inbfrq 0

mini sd nstep 200

set nstep = {nstep} 
set temp = {temperature.value_in_unit(unit.kelvin)}

scalar fbeta set 5. sele all end  
open write unit 21 file name lig_in_vacuum.dcd
 
DYNA lang leap start time 0.001 nstep @nstep -
    nprint {nstout} iprfrq {nstout} -
    iunread -1 iunwri -1 iuncrd 21 iunvel -1 kunit -1 -
    nsavc {nstdcd} nsavv 0 -
    rbuf 0. tbath @temp ilbfrq 0  firstt @temp -
    ECHECK 0
    
stop"""
        return body

    def _get_CHARMM_waterbox_postprocessing_body(self):
        ##### solv phase ######
        switch = self.vdw_switching_keyword

        body = f"""
stream charmm_step3_pbcsetup.str

!
! Image Setup
!

open read unit 10 card name charmm_crystal_image.str
CRYSTAL DEFINE @XTLtype @A @B @C @alpha @beta @gamma
CRYSTAL READ UNIT 10 CARD

!Image centering by residue
!IMAGE BYRESID XCEN @xcen YCEN @ycen ZCEN @zcen sele resname TIP3 end

!
! Nonbonded Options
!
! cons fix sele segi solv end

nbonds atom vatom {switch} bycb -
       ctonnb 10.0 ctofnb 12.0 cutnb 12.0 cutim 12.0 -
       inbfrq 1 imgfrq 1 wmin 1.0 cdie eps 1.0 -
       ewald pmew fftx @fftx ffty @ffty fftz @fftz  kappa .34 spline order 6

energy

!
!use a restraint to place center of mass of the molecules near the origin
!

open read file unit 41 name ../traj.dcd
traj query unit 41

set start ?start
set nframes ?nfile
set skip ?skip

set nframes @nframes !?nfile
traj firstu 41 nunit 1 begi @start skip @skip stop @nframes

open form writ unit 51 name ener_solv.log
echu 51
set idx 0
label loop
traj read
energy
echo ?ener
incr idx by 1
if @idx .lt. @nframes goto loop
        
  
stop"""
        return body

    def _get_CHARMM_waterbox_production_body(self):
        ##### waterbox ######
        switch = self.vdw_switching_keyword

        nstep = self.configuration["simulation"]["parameters"]["nstep"]
        nstout = self.configuration["simulation"]["parameters"]["nstout"]
        nstdcd = self.configuration["simulation"]["parameters"]["nstdcd"]
        if self.configuration["simulation"].get("GPU", False) == True:
            GPU = f"""domdec gpu only"""
        else:
            GPU = ""

        body = f"""
!
! Setup PBC (Periodic Boundary Condition)
!

stream charmm_step3_pbcsetup.str

!
! Image Setup
!

open read unit 10 card name charmm_crystal_image.str
CRYSTAL DEFINE @XTLtype @A @B @C @alpha @beta @gamma
CRYSTAL READ UNIT 10 CARD

!Image centering by residue
!IMAGE BYRESID XCEN @xcen YCEN @ycen ZCEN @zcen sele resname TIP3 end

!
! Nonbonded Options
!

nbonds atom vatom {switch} bycb -
       ctonnb 10.0 ctofnb 12.0 cutnb 16.0 cutim 16.0 -
       inbfrq -1 imgfrq -1 wmin 1.0 cdie eps 1.0 -
       ewald pmew fftx @fftx ffty @ffty fftz @fftz  kappa .34 spline order 6

energy

!
!use a restraint to place center of mass of the molecules near the origin
!

MMFP
GEO rcm sphere -
    Xref @xcen Yref @ycen Zref @zcen XDIR 1.0 YDIR 1.0 ZDIR 1.0 -
    harmonic FORCE 1.0 select .not. ( hydrogen .or. resname TIP3 ) end
END

shak bonh para fast sele segi SOLV end

mini SD nstep 500
mini ABNR nstep 500

!
! NPT dynamics:
! you can change
! nstep  : number of MD steps
! nprint : print-out frequency
! nsavc  : the trajectory saving frequency
!

! estimate Pmass from SYSmass (total system mass)
! [there could be problems with exreme values, such as  Pmass << SYSmass or Pmass >> SYSmass
scalar mass stat
calc Pmass = int ( ?stot  /  50.0 )

energy
{GPU}
energy

set nstep = {nstep}
set temp = {temperature.value_in_unit(unit.kelvin)}

scalar fbeta set 5. sele all end 
open write unit 13 file name lig_in_waterbox.dcd 

DYNA CPT leap start time 0.001 nstep @nstep -
     nprint {nstout} iprfrq {nstout} -
     iunread -1 iunwri -1 iuncrd 13 iunvel -1 kunit -1 -
     nsavc {nstdcd} nsavv 0 -
     PCONSTANT pref   1.0  pmass @Pmass  pgamma   20.0 -
     lang rbuf 0. tbath @temp ilbfrq 0  firstt @temp -
     ECHECK 0

stop"""
        return body

    def _get_CHARMM_postprocessing_header(self, env: str) -> str:

        intermediate_filename = self.configuration["system"][self.structure][env][
            "intermediate-filename"
        ]

        date = datetime.date.today()
        header = f"""*Version from {date} 
*Run script for CHARMM jobs from transformato 
*

! Read topology and parameter files 
stream charmm_toppar.str 

! Read PSF 
open read unit 10 card name {intermediate_filename}.psf 
read psf  unit 10 card

! Read Coordinate 
open read unit 10 card name {intermediate_filename}.crd 
read coor unit 10 card"""
        return header
