from win32com.client import CastTo, Dispatch
from win32api import GetFileVersionInfo, LOWORD, HIWORD
# import csv
import json
import os
import time
import numpy as np
from matplotlib import pyplot as plt
import AnaBits
import FileManagement
import InstrSwitch
import InstrAWG
import ScopeControl
import InstrSourceMeter
import InstrSMU
import UtilsCollection
import Pub
import WFGeneration

## TEO not working? looki here and erase. C:\Users\User\AppData\Local\Temp\gen_py\3.6

DEBUG, SHOWINFO, DELTAT = True, False, 2
TEOCHUNK = 2048

def CheckTEOCasting(to, name):
    """
    :param to:      where it is cast to
    :param name:    name of TEO's item
    :return:        result
    from Tyler
    """
    # CastTo that prints a warning if it doesn't work for some reason
    result = None
    if name == "ITS_DriverIdentity":
        try:
            result = CastTo(to, name)
        except AttributeError:
            UtilsCollection.WriteLog(None, ["This is (most likely) a TEO license error - 408-332-4449"], Pub.CRASHINGMSG)
    else:
        result = CastTo(to, name)
    if result is None: UtilsCollection.WriteLog(None, [f"{name} has failed"], Pub.CRASHINGMSG)
    return result

def DecodeSenseRs(SenseRList: list):
    """
    :param SenseRList:        List from General file (3 elements)
    :return: SenseR = value of sense R used,  IsHiSenseR = is it the high one?
    """
    SenseR = SenseRList[0]
    IsHiSenseR = SenseRList[0] == SenseRList[2]
    if SenseRList[1] >= SenseRList[2]: UtilsCollection.WriteLog(None, ["2nd resistor value MUST be < than third [first is selected]"], Pub.CRASHINGMSG)
    return SenseR, IsHiSenseR

def DeleteWFIDs():
    """
    :return:
    To avoid confusion: delete all TEO WFIDs. to be called producrion test
    """

    Fop = FileManagement.FileOp("C")
    Fop.Path = Pub.PATH4WFIDS
    WFIDs = Fop.listfiles(oftype='.JSON', withpath=True)
    for ff in WFIDs:
        os.remove(ff)
    return

def GenerateTestWaveform():
    """
    :return:        A, Ampl
    derived from some VTgh set of parameetrs
    """
    paX = {}
    paX["Pol"] = 1
    paX["Start_V"] = 0.5
    paX["End_V"] = 8
    paX["Step_V"] = 0.1
    paX["Deltat_ns"] = DELTAT
    paX["Fall_ns"] = 10
    paX["Dly_ns"] = 900
    paX["Rise_ns"] = 10
    paX["Dur_ns"] = 100
    paX["Fall_ns"] = 10
    paX["PreFire_V"] = 0
    paX["OSAVar"] = 0.0
    paX["OSVarTau_ns"] = 30
    paX["OSVarDur_ns"] = 10
    paX["OSA"] = 0
    paX["OSDur_ns"] = 0
    paX["OSTau_ns"] = 10
    paX["TONCtime_ns"] = 0
    paX["TONCampl"] = -1
    ww = WFGeneration.GenerateWFVT(paX)
    t, A, _, _ = ww.CreateVTSeq()
    A = np.asarray(A, dtype=float)
    return A, paX["End_V"]

def GetPoints4TEOsDcMeasurement(Meastime=0.1):
    """
    :param Meastime:        in seconds
    :return:  # of pointsof WF and how many one should evaluate to reduce 60Hz noise
    """
    FNET, DCPERIOD= 60, 3.2e-5
    nPtsWF = int(min([Meastime / DCPERIOD, 8000]))                                                                                                  # points in WF, limit is 8000
    PtsPerPeriod = 1 / FNET / DCPERIOD
    PtsEVal = int(PtsPerPeriod * (np.floor(nPtsWF / PtsPerPeriod) - 1))                                                                             # cut off 1 cycle at end
    return nPtsWF, PtsEVal

def GlitchTest(Vm, WFinfo):
    """
    :param Vm:              1D aray with voltages for checking
    :param WFinfo:          if not None: list with Period in index land and # of pulses
    :return:                Flag: allOk
    TEO can have glitches (if so investigate USB). thisd is a test to find teh splots where it does so.
    """
    allOk, MARG, GLIM, GlLocs = True, 50, 0.1, []
    if WFinfo is None:  return allOk, GlLocs

    SHOWGLITCHTEST = True

    nPts = Vm.size
    nPtsReq = WFinfo[0] * WFinfo[1]
    if nPtsReq + MARG * 2 >= nPts:  raise ValueError("GlitchTest in TEObox: not enough points - ups.")
    WFmin, WFmax = np.min(Vm), np.max(Vm)
    Pol = 0
    if abs(WFmin) > abs(WFmax) / 2:
        Pol = -1
    elif abs(WFmax) > abs(WFmin / 2):
        Pol = 1
    if Pol > 0:  # find last edge
        idxLE = np.where(Vm > WFmax / 2)[0][-1]
    elif Pol < 0:
        idxLE = np.where(Vm < WFmin / 2)[0][-1]
    else:
        idxLE = max[np.where(Vm > WFmax / 2)[0][-1], np.where(Vm < WFmin / 2)[0][-1]]

    i2 = idxLE + MARG
    i1 = idxLE - nPtsReq + MARG
    Curs = np.reshape(Vm[i1:i2], (WFinfo[1], WFinfo[0])).T
    Diffs = np.zeros((WFinfo[0], WFinfo[1]))
    GlLocs, GlPuNo = [], []
    for i in range(WFinfo[1]):
        if i == 0:
            Diffs[:, i] = Curs[:, i + 1] - Curs[:, 0]
        elif i == WFinfo[1] - 1:
            Diffs[:, i] = Curs[:, i] - Curs[:, -2]
        else:
            Diffs[:, i] = (Curs[:, i - 1] + Curs[:, i + 1]) / 2 - Curs[:, i]

        idxD = np.where(np.abs(Diffs[:, i]) > GLIM)[0]
        if len(idxD) > 0:
            idxD1 = np.where(np.abs((Curs[:, i - 1] - Curs[:, i])) > GLIM)[0]
            idxD2 = np.where(np.abs((Curs[:, i + 1] - Curs[:, i])) > GLIM)[0]
            if len(idxD1) > 0 and len(idxD2) > 0:  # avoid double counting
                GlLocs.append(idxD)
                GlPuNo.append(i)

    if len(GlPuNo) > 0:
        allOk = False
        if SHOWGLITCHTEST:  # shit have glitch
            t = np.linspace(0, WFinfo[0] - 1, WFinfo[0])
            for gg in range(len(GlPuNo)):
                plt.plot(t, Curs[:, GlPuNo[gg]], 'k')
                if 0 < GlPuNo[gg] < WFinfo[1]:
                    plt.plot(t[GlLocs[gg]], Curs[GlLocs[gg], GlPuNo[gg]], 'ro')
                else:
                    plt.plot(t[GlLocs[gg]], Curs[GlLocs[gg], GlPuNo[gg]], 'bo')

                GlLocs[gg] = GlLocs[gg] + i1 + GlPuNo[gg] * WFinfo[0]
                Glitches = "Glitches " + str(GlLocs[gg])
                plt.annotate(Glitches, xy=(0.05, 0.95 - gg * 0.06), xycoords="axes fraction", fontsize=7)

                print(GlLocs[gg])
            plt.show()
    return allOk, GlLocs

def NoiseandSat(pa, MeasType):
    """
    :param pa:          parameter  dict with keys General, Aux
    :param MeasType:    Mnemonic for measurement, such as VTh = key name
    :return:    Noise per point in µA, saturation current in A, averages in pulse
    For fully internal TEO. Looks up the numbers in the parameter set when TEO box is not available
    """
    Pref, NoisepPt, Sat, Aves = "", None, None, None
    if "Setup" in pa.keys() and not isinstance(pa["Setup"], str):
        paM, paSc = pa["Setup"][MeasType], pa["Setup"]["Scope"]
    else:
        if "Scope" in pa.keys(): paSc = pa["Scope"]
        paM = pa[MeasType]

    if MeasType == "fOTS":  Pref = "VTh_"

    if MeasType in ["VTh", "fOM", "fOTS", "Dr", "fS"]:                                                                                              # Dr/fD can be called by drift suite
        Aves = int(round((paM[Pref + "Dur_ns"] - paM[Pref + "EvalStart_ns"] - paM[Pref + "EvalEnd_ns"]) / paSc["Ts"] / 1e9))
    elif MeasType in ["OP", "PCM"]:
        Aves = None
    elif MeasType == "Lk":
        return NoisepPt, Sat, Aves
    elif MeasType == "Adv":                                                                                                                         # some analyes of advanced can use this too
        if "AdvTest" in pa["Setup"]["Adv"] and pa["Setup"]["Adv"]["AdvTest"] == "Dr":
            Aves = int(round((paM[Pref + "Dur_ns"] - paM[Pref + "EvalStart_ns"] - paM[Pref + "EvalEnd_ns"]) / paSc["Ts"] / 1e9))
    else:
        raise ValueError("need to add " + MeasType + " in TEOModule.NoiseandSat")

    GaindB = pa["Aux"]["TEO"]["TEOGain0"]
    if paM["TEOgain"] != -99 and paM["TEOgain"] != pa["Aux"]["TEO"]["TEOGain0"]: GaindB = paM["TEOgain"]                                            # update GaindB
    Sat = pa["Aux"]["Board"]["Sat1"]
    NoisepPt = pa["Aux"]["Board"]["NoisepPt"]
    if pa["General"]["TEOintHF"] == 1:
        idxLo = 3                                                                                                                                       # order :0 = highest, then down
        if GaindB > 22: idxLo = int(26 - GaindB)                                                                                                        # gain index of lim BW amp
        _, isHiR = DecodeSenseRs(pa["General"]["TEOSenseR"])
        if isHiR:
            Sat = pa["Aux"]["TEO"]["SatHiR"][idxLo]
            NoisepPt = pa["Aux"]["TEO"]["NoisepPtHiR"][idxLo]
        else:
            Sat = pa["Aux"]["TEO"]["SatLoR"][idxLo]
            NoisepPt = pa["Aux"]["TEO"]["NoisepPtLoR"][idxLo]
    return NoisepPt, Sat, Aves

class TEObox(object):
    """
    Container for TEO control. Talks also to all other devices
    """
    def __init__(self, paX, InstList, **Instrct):
        """
        :param      paX:        parameters. MUST have "General" key to initialize
        :param      InstList:   list of external devices that need to be instantiated. Caller to provide
        **Instrt:     additional instructions
        ActInstr:           interact with instruments (set to false when analyzing things)
        IsCalibration:      called by calibration
        1. determine how we use TEO
        2. initialze com objects and establish communications
        The clue of this is to use the TEObox in a somewhat similar way as the old Sc. Inside the TEOBox, we keep the TEO instance open to save
        time. Also, inside this box, we make instantiations to all other devices (Scope, AWG, etc) as needed. the caller must provide a list of
        POTENTIAL devices. the Teobox applies its logic ti figure out whether they are actually needed. These are kept alive as well to save time.
        """
        ActInstr, IsCalibration = True, False
        if "ActInstr" in Instrct:           ActInstr = Instrct["ActInstr"]
        if "IsCalibration" in Instrct:      IsCalibration = Instrct["IsCalibration"]

        self.MAGICLENGTHS = InstrAWG.MAGICLENGTHS                                                                                                   # can use TEO's in AWG mode only, compatible to Rigol
        self.pa = paX
        self.TEOProgDur, self.TEOTS = None, 2 / 1e9
        self.TEOactive = "TEO" in self.pa["General"].keys() and self.pa["General"]["TEO"] == 1                                                      # if False, TEO is bypassed

        self.TEOintHF = self.pa["General"]["TEOintHF"] == 1                                                                                         # do we use TEO's AWG and scope?
        self.FullTEO = self.TEOintHF                                                                                                                # convenience..
        self.TEOextHF = self.pa["General"]["TEOextHF"] == 1                                                                                         # do we use external HF things?
        self.TEOintLF = self.pa["General"]["TEOintLF"] == 1                                                                                         # do we use TEO's internal LF mode?
        self.HalfTEO = self.pa["General"]["TEOintHF"] == 0.5                                                                                        # 'half' TEO = AWG only

        self.TEOSwRev = None
        if "TEOSoftwareRev" in self.pa["General"]: self.TEOSwRev = int(float(self.pa["General"]["TEOSoftwareRev"]))                                 # TEO driver ID is useless. This way we can manually keep track on his revs
        if self.TEOintHF and self.HalfTEO:  UtilsCollection.WriteLog(None, ["General setup: TEO mode for internal HF not valid"], Pub.CRASHINGMSG)
        """
        Note for the gain mess: one setting, two channels. We stick with the traditional gain settings between -5 and 26 to maintain continuity. The
        defs here calculate the gains for the high BW and low BW separately, using TEO's steps.
        """
        self.BWCompatibility()
        self.HFgain = float(self.pa["Aux"]["TEO"]["TEOGain0"])                                                                                      # gain for calibration, controls saturation, define this as full BW
        self.HFgainLoBW, self.PROBECARDREV = None, None                                                                                             # clean up the gain mess
        self.invCh = self.pa["General"]["TEOInvCh"]                                                                                                 # which channel the inverted PS output goes to

        self.DriverIdentity, self.DeviceIdentity, self.DeviceControl = None, None, None
        self.LF_Measurement, self.HF_Measurement, self.LF_Measure, self.HF_Gain, self.HF_Input = None, None, None, None, None
        self.LF_Voltage, self.AWG_WaveformManager, self.wf, self.LFMode, self.LFOffset = None, None, None, None, None
        self.DevName, self.DevDesc, self.DevSN, self.DevRev = None, None, None, None
        self.LFVolt, self.LFVmin, self.LFVmax, self.LFNpts, self.LFtime, self.DISCARD = None, 0, 1, 11, 0.2, 200

        self.TrcFolder, self.RecIDwPathJSON, self.RecIDwPathBIN, self.RecID = None, None, None, None
        self.Sw, self.Sc = None, None
        self.TEOattndB = 14
        self.TEOAttn = 10 ** (self.TEOattndB/20)
        self.NCycles, self.BeforeTrig, self.AfterTrig = 1, None, None                                                                               # for HalfTEO and TEO
        self.ATLen2Read, self.AT, self.ATLen = None, None, None

        # 2 establish communication to TEO
        if self.TEOactive and ActInstr: self.InitializeTEO()

        # 3 instantiate other devices if needed. Logic: let it go opposite to internal HF. If ExtHF is used, it will be fast and we need a LC.
        Msg = "Wrong parameters in test system. When using picoscope, the TEOintHF must not be 1"
        if not IsCalibration:
            if self.TEOintHF and "Sc" in InstList: UtilsCollection.WriteLog(None, [Msg], Pub.CRASHINGMSG)
        if ActInstr and len(InstList) > 0:
            if "Sc" in InstList:
                paSc = {"General": paX["General"], "Aux": paX["Aux"], "Scope": {"Ch2Reverse": self.invCh}}                                          # only 1 -> use D
                paSc["Scope"]["Cal"] = paX["Aux"]["Board"]["Cal"]
                self.Sc = ScopeControl.Scope(paSc)

            if "AWG" in InstList:
                if "Common" not in paX.keys():                                                                                                      # can be called at init wo Common Key. Set this so for now.
                    paX["Common"] = {"NCycl": 1}
                elif "NCycl" not in paX["Common"]:
                    paX["Common"]["NCycl"] = 1

                if self.HalfTEO:
                    self.NCycles = paX["Common"]["NCycl"]
                    self.HF_Measurement.SetHF_Mode(0, False)
                else:
                    paAWG = {"Addr": paX["General"]["AWGaddress"], "DefCh": paX["General"]["DefCh"], "ExpResp": paX["General"]["AWGExpResp"], "NCycl": \
                        paX["Common"]["NCycl"]}
                    # This is the trick to use the other channel to force the signal to 0 (SetTo0=False => use default channel = true)
                    paAWG["useDefCh"] = True                                                                                                        # just in case
                    #self.HF_Measurement.SetHF_Mode(0, False)
                    self.Aw = InstrAWG.AWGclass(paAWG)

            if "Sm" in InstList:
                # can initialize these, because we won't change.
                paSMU = {"Addr": self.pa["General"]["SMUaddress"], "Mode": self.pa["Lk"]["Mode"], "Range": self.pa["Lk"]["Range"]}
                paSMU["ExpResp"] = self.pa["General"]["SMUExpResp"]
                paSMU["ACQDly"] = self.pa["Lk"]["ACQDly"]
                paSMU["nReads"] = self.pa["Lk"]["nReads"]
                paSMU["KSstairs"] = self.pa["Lk"]["KSstairs"]
                self.Sm = InstrSMU.SMUclass(paSMU)

            if "SmK" in InstList:
                paKy = {"Addr": self.pa["General"]["SRCMeteraddress"], "ExpResp": self.pa["General"]["SRCMeterExpResp"]}
                self.SmK = InstrSourceMeter.SrcMtrclass(paKy)

        if self.pa["General"]["TEO"] != 1: return

        # Section for the rev 2 board
        self.SenseR, self.HiSenseR = DecodeSenseRs(self.pa["General"]["TEOSenseR"])
        self.CALFILEREV2 = "C:\\NVMPyMeas\\Setups\\TEORev2Cal.JSON"
        self.CurOffsetHiR = [0, 0, 0, 0]                                                                                                            # 4 numbers for the 4 gains of the limBW amplifier
        self.CurOffsetLoR = [0, 0, 0, 0]                                                                                                            # 4 numbers for the 4 gains of the limBW amplifier
        self.CurCalHiR = [0.0000738007, 1, 1, 1]                                                                                                    # 4 numbers for the 4 gains of the limBW amplifier
        self.CurCalLoR = [1, 1, 1, 1]                                                                                                               # 4 numbers for the 4 gains of the limBW amplifier
        self.NoisepPtHiR, self.NoisepPtLoR = [1] * 4, [1] * 4
        self.VMonOffSet = 0                                                                                                                         # 1 number for voltage monitor (internal)
        self.VMonCal = 1                                                                                                                            # 1 number for voltage monitor (internal)
        self.CurCalX = 1                                                                                                                            # 1 number for the external amplifier gain limited BW
        self.CurCalFullX = 1                                                                                                                        # 1 number for the external amplifier gain, high BW
        self.VMonCalX = self.pa["Aux"]["TEO"]["VMonCalX"]                                                                                           # 1 number for voltage monitor (external)
        self.CalMeasured = 0
        self.LineDly_ns = 0
        self.LineDlyX_ns = 0
        self.LineDlyXFull_ns = 0
        self.TEOShift = 12                                                                                                                          # there is some shift between the ref and the (voltage) data
        self.CalItms = ["CurOffsetHiR", "CurOffsetLoR", "CurCalHiR", "CurCalLoR", "VMonOffset", "VMonCal",                                          # skip offsets for external outputs.
                        "CurCalHiRX", "CurCalLoRX", "CurCalHiRFullX", "CurCalLoRFullX", "NoisepPtLoR", "NoisepPtHiR",
                        "VMonCalX", "CalMeasured", "TEOShift", "LineDlyHiR_ns", "LineDlyLoR_ns", "LineDlyX_ns", "LineDlyXFull_ns"]
        self.CurOffSetUse, self.CurCalUse, self.NoisepPtUse = None, None, None
        self.CurCalX, self.CurCalXFull = None, None                                                                                                 # full BW channel is calculated throughout
        self.LineDly, self.LineDlyX, self.LineDlyXFull = None, None, None
        self.TEOZEROMARG = 10                                                                                                                       # keep a disstance of 10 samples from start of 1st pulse => for local zero correction
        self.SelectedGain, self.Setupis4Rev2, self.SaturationUse = None, None, None
        self.GetCalibrationData()
        self.SelectCalibration(self.HFgain, IsCalibration=IsCalibration)
        self.WFGlitchFree, self.LastProgedWFName = True, ""                                                                                         # Keep this on True. Can switch to false if one wants to look for glitches
        if SHOWINFO: print("end of INIT")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # do NOT close the instance to the picoscope - still need it
        #print("exit")
        pass

    def AdjustMainGain(self, NewGaindB):
        """
        :param          NewGaindB:  refers to gain (NOT steps) for the HiBW amplifier as in the initial version.
        :return:
        """
        if not self.TEOactive: return
        self.HFGainGetValues()
        if self.HFgain != NewGaindB and NewGaindB != -99:
            self.SetHFGain(NewGaindB)
        self.SelectCalibration(self.HFgain)

    def AlignbyV(self, EL, x, y):
        """
        :param EL:      length (at end) where sync pulse will occur
        :param x:       entire voltage trace
        :param y:       entire current trace
        :return: Aligned and shortened arrays, Step excluding prefix of VMon data. Aligned (with reference that is). The line delay for the current is considered as well.
        1. Issue is that x can come with a huge offset. The first thing is to find out the polarity by looking at the Ref
        2. Correct the offset of VMon and make the pulse +
        3. Normalize VMon
        4. Use Canny - but be mindful that canny peak of the reference is in center.... Assign middle
        5. Roll the voltage array as indicated
        6. Roll the current array - take additional shift due to line delay into account
        IMPORTANT: Gates: it can handle a gate which is at the start. Implemented for drift.
        """
        Pol, Shift = 1, None                                                                                                                        # simplify: turn around
        EDGE = int(WFGeneration.SYNCLEN / 2)
        if np.mean(self.AT[-EL:]) < 0: Pol = -1
        Ref = Pol * self.AT[-EL:]                                                                                                                   # coarse aligmnent
        RefMax = np.max(Ref)
        idx = np.where(Ref > RefMax / 2)[0][-1]                                                                                                     # last one!

        i1 = self.ATLen - EL + idx - int(1.7 * WFGeneration.SYNCLEN)                                                                                # make that longer to be able to find zero foffset
        i2 = i1 + int(2.4 * WFGeneration.SYNCLEN)
        Ref = Pol * self.AT[i1:i2]
        Ref = Ref / np.max(Ref)                                                                                                                     # offset correction
        Cut = self.ATLen - self.ATLen2Read
        VMon = (x[i1 - self.TEOShift - Cut:i2 - self.TEOShift - Cut])                                                                               # need to correct for what the gates have cut
        Offset = [np.mean(VMon[:EDGE]), np.mean(VMon[-EDGE:])]
        if Pol == 1:    Offset = min(Offset)
        else:           Offset = max(Offset)
        VMon = (VMon - Offset) * Pol

        Norm = np.max(VMon)                                                                                                                         # normalize
        idx = np.where(VMon > 0.95 * Norm)[0]
        Norm = np.mean(VMon[idx])
        VMon = VMon / Norm

        Canny = AnaBits.ConstructCanny(VMon, Si=1)
        CannyR = AnaBits.ConstructCanny(Ref, Si=1)
        idxMaxR = np.where(CannyR > 0.9)[0][0]
        idxMinR = np.where(CannyR < -0.9)[0][0]

        idxMax, idxMin = np.argmax(Canny), np.argmin(Canny)
        if idxMaxR - idxMax == idxMinR - idxMin:
            Shift = idxMaxR - idxMax + self.TEOShift
        else:
            plt.plot(Ref, 'k')
            Sc = np.max(Ref) / np.max(VMon)
            plt.plot(VMon * Sc, 'b')
            plt.plot(np.abs(CannyR), 'b--')
            plt.plot(np.abs(Canny) * Sc, 'g')
            plt.title("Problem")
            plt.show()
        x = np.roll(x, Shift)                                                                                                                       # positive shifts to left
        y = np.roll(y, Shift - self.LineDly)
        return x, y, Pol, Norm

    def AlignbyVdepricate(self, EL, x, y):
        """
        :param EL:      length (at end) where sync pulse will occur
        :param x:       entire voltage trace
        :param y:       entire current trace trace
        :return: Aligned and shortened arrays, xmax of Vmon data. Aligned (with reference that is). The line delay for the current is considered as well.
        1. Find the last falling edge of the ref and so we know where the sync pulse sits. Be mindful of polarity - make all +
        2. Correlate 1.4 times the interesting section between ref and voltage trace
        3. Roll the voltage array as indicated
        4. Roll the current array - take additional shift due to line delay into account
        """
        Pol = 1                                                                                                                                     # simplify: turn around
        if np.mean(self.AT[-EL:]) < 0: Pol = -1
        Ref = Pol * self.AT[-EL:]
        RefMax = np.max(Ref)
        idx = np.where(Ref > RefMax / 2)[0][-1]                                                                                                     # last one!
        i1 = self.ATLen - EL + idx - int(3 * WFGeneration.SYNCLEN)
        i2 = i1 + int(5 * WFGeneration.SYNCLEN)
        Ref = Pol * self.AT[i1:i2]# - np.mean(self.AT[i1:i2])
        #MonOff = np.mean(x[i1 - self.TEOShift:i2 - self.TEOShift])
        VMon = (x[i1 - self.TEOShift:i2 - self.TEOShift]) * Pol
        xMax = np.max(VMon)
        LRef = len(Ref)
        Shift = np.argmax(np.correlate(Ref, VMon, "full")) - (LRef - 1) + self.TEOShift
        x = np.roll(x, Shift)                                                                                                                       # positive shifts to left
        y = np.roll(y, Shift - self.LineDly)
        # TODO back to 1.2/1.4 and zerocorr
        return x, y, Pol, xMax

    def BWCompatibility(self):
        """
        :return:
        when the last TEO board was introduced. the AuxDict was changed with the middle key called TEO. Need to fix
        """
        if "TEO" in self.pa["Aux"]: return                                                                                                          # is new - nothing to do
        self.pa["Aux"]["TEO"] = {"TEOGain0": self.pa["Aux"]["TEOGain0"], "TEOVMax": self.pa["Aux"]["TEOVMax"]}                                      # move keys and create the new structure

    def CheckGlitchFreeness(self, A, Ampl, CallerKey, WFinfo):
        """
        :param A:           WF array normalized
        :param Ampl:        calibration corrected amplitude
        :param CallerKey:   = WF name
        :param WFinfo:      list with period and # pulses
        This rewrites until the glitch test no longer finds glitches. Can presumably be sped up by re-writing individual points. Is not necessary if
        the USB connection is good. Leave in case we need it again.
        :return:
        """
        Cnt, CNTMAX = 0, 50
        if self.WFGlitchFree: return

        allOk = False
        while not allOk and Cnt < CNTMAX:
            Cnt = Cnt + 1
            d, Status = self.MeasureWFs()
            allOk, GlitchList = GlitchTest(d[:, 0, 0], WFinfo)                                                                                                  # suffices to test just one voltage
            if not allOk:
                Fname2Erase = Pub.PATH4WFIDS + "\\" + self.LastProgedWFName + ".JSON"
                if os.path.isfile(Fname2Erase): os.remove(Fname2Erase)
                self.ProgramAWG(A, Ampl, CallerKey, True)
                # for ggL in range(len(GlitchList)):
                #     for gg in GlitchList[ggL]:
                #         self.wf.AddSample(gg, self.AT[gg], self.Trig1[gg], self.Trig2[gg])
                self.AWG_WaveformManager.Run(CallerKey, int(self.NCycles))
                d, Status = self.MeasureWFs()
                allOk, GlitchList = GlitchTest(d[:, 0, 0], WFinfo)

        if not allOk:
            #raise ValueError("Glitches cannot be removed")
            print("cannot fix")
        else:
            self.WFGlitchFree = True
            print("OK - no glitches")

    def CreateCalFile(self, CDat=None):
        """
        :param CDat:            dict with calibration data. If None it creates a file so that therte is something
        :return:
        This is intended for internal measuremenmts only
        """

        if CDat is None:
            CDat = {}
            for ii in self.CalItms:
                x = getattr(self, ii)
                CDat[ii] = x
        UtilsCollection.WriteDictJson(self.CALFILEREV2, CDat)
        return

    def DefineRecID(self, TrcFolder, RecID):
        """
        :param TrcFolder:           Folder for trace files
        :param RecID:               RecID as defined by e.g. Prod-test
        :return:
        This used to be in ScopeControl and has been moved to the TEOBox. The TEOBox passes it on if the picoscope if being used. The PS needs this
        to put the data into the correct place
        """

        self.TrcFolder = TrcFolder
        if os.path.isdir(TrcFolder):
            self.RecIDwPathBIN = TrcFolder + "\\" + RecID + ".bin"
            self.RecIDwPathJSON = TrcFolder + "\\" + RecID + ".JSON"
            self.RecID = RecID
        else:
            UtilsCollection.WriteLog(self.pa["General"]["DataDrive"], ["Trace folder does not exist"], Pub.CRASHINGMSG)

        if self.Sc is not None and isinstance(self.Sc, ScopeControl.Scope):
            self.Sc.DefineRecID(TrcFolder, RecID, self.RecIDwPathBIN, self.RecIDwPathJSON)                                                          # Picoscope: communicates only the paths & names

    def DisplayTEOMode(self):
        """
        :return:
        Quick message so that one can see how it is set
        """
        ModeH, ModeL = "Rigol", "LF: ext Instr (KeySight)"
        if self.TEOintHF:
            ModeH = "internal HF (full)"
        elif self.HalfTEO:
            ModeH = "AWG internal, picoscope"
        if self.TEOextHF: ModeH = ModeH + " external HF source"

        if self.TEOintLF: ModeL = "internal dc mode"
        print("initializing TEO. Mode = " + ModeH + " , " + ModeL)

    def GetCalibrationData(self):
        """
        :return:
        It reads the calibration file for the rev2 board. The whole thing is geared towards the internal reading.
        There are the following sets of calibration data:
        CurOffSetHiR:   4 numbers for the 4 gains of the limBW amplifier
        CurOffSetLoR:   4 numbers for the 4 gains of the limBW amplifier
        CurCalHiR:      4 numbers for the 4 gains of the limBW amplifier
        CurCalLoR:      4 numbers for the 4 gains of the limBW amplifier
        VMonOffSet:     1 number for voltage monitor (internal)
        VMonCal:        1 number for voltage monitor (internal)
        CurOffSetHiRX:  4 numbers for the 4 gains of the limBW amplifier
        CurOffSetLoRX:  4 numbers for the 4 gains of the limBW amplifier
        CurCalHiRX:     4 numbers for the 4 gains of the limBW amplifier
        CurCalLoRX:     4 numbers for the 4 gains of the limBW amplifier
        CurOffSetHiRFullX: 4 numbers for the corresponding gains of the fullBW amplifier (external only)
        CurCalHiRFullX: 4 numbers for the corresponding gains of the fullBW amplifier (external only)
        VMonOffSetX:    1 number for voltage monitor (internal)
        VMonCalX:       1 number for voltage monitor (internal)

        If the calibration data are not available, it gives a warning and proceeds with cal=1 and offset=0.
        """
        if not os.path.isfile(self.CALFILEREV2):
            self.CreateCalFile()
        CalDat = UtilsCollection.ReadDictJson(self.CALFILEREV2)
        for ii in self.CalItms:
            setattr(self, ii, CalDat[ii])
        if self.CalMeasured == 0: print("Warning: you do not have a calibration file for fully internal TEO operation. Ignore if you don't use that.")
        return

    def GetLFOffset(self):
        """
        :return:        see comments at LFcurrent
        """
        NOISEEST = False
        Noff = 10
        if NOISEEST: Noff = 100
        self.LF_Voltage.SetValue(0)
        CurObj = self.LF_Measurement.LF_MeasureCurrent(self.LFtime)
        Cur = CurObj.GetWaveformDataArray()
        Offset = [np.mean(Cur[self.DISCARD:])]
        if self.LFOffset is None:
            for i in range(Noff):
                CurObj = self.LF_Measurement.LF_MeasureCurrent(self.LFtime)
                Cur = CurObj.GetWaveformDataArray()
                Offset.append(np.mean(Cur[self.DISCARD:]))
        Offset = np.asarray(Offset)
        self.LFOffset = np.mean(Offset)
        if NOISEEST: print("offset " + str(self.LFOffset) + " sigma " + str(np.std(Offset)))

    def GetTEOsSWVersion(self):
        """
        Read the version number. TEO is always at the same spot.
        https://www.blog.pythonlibrary.org/2014/10/23/pywin32-how-to-get-an-applications-version-number/
        :return:
        """
        info = GetFileVersionInfo("C:\\Program Files (x86)\\TS_Memory_Tester\\TSX_MT_MemoryTester.exe", "\\")
        #ms = info['FileVersionMS']
        ls = info['FileVersionLS']
        self.TEOSwRev = HIWORD(ls) + LOWORD(ls) / 10

    def InitializeTEO(self):
        """
        :return:        all in self
        Talk to TEO and get the com objects
        """
        self.DisplayTEOMode()
        self.GetTEOsSWVersion()
        HMan = Dispatch("TSX_HMan")
        if HMan is None: UtilsCollection.WriteLog(None, ["TSX_HMan has failed"], Pub.CRASHINGMSG)
        self.DriverIdentity = CastTo(HMan.GetSystem("MEMORY_TESTER"), "ITS_DriverIdentity")
        self.DeviceIdentity = CheckTEOCasting(self.DriverIdentity, "ITS_DeviceIdentity")
        self.DeviceControl = CheckTEOCasting(self.DriverIdentity, "ITS_DeviceControl")
        self.LF_Measurement = CheckTEOCasting(self.DriverIdentity, "ITS_LF_Measurement")
        self.HF_Measurement = CheckTEOCasting(self.DriverIdentity, "ITS_HF_Measurement")
        self.HF_Gain = CheckTEOCasting(self.HF_Measurement.HF_Gain, "ITS_DAC_Control")
        self.AWG_WaveformManager = CheckTEOCasting(self.HF_Measurement.WaveformManager, "ITS_AWG_WaveformManager")

        self.DeviceControl.StartDevice()

        #  === find out current state, DO FOR DEVICE AND SWITCHING = LFMode ========================================================================
        self.DevName = self.DeviceIdentity.GetDeviceName()
        #self.DevDesc = self.DeviceIdentity.GetDeviceDescription() # not very useful
        self.DevRev = [self.DeviceIdentity.GetDeviceMajorRevision(), self.DeviceIdentity.GetDeviceMinorRevision()]
        self.DevSN = self.DeviceIdentity.GetDeviceSerialNumber()
        if self.DevSN != self.pa["General"]["TEOExpResp"]:
            Msg = "Expected response = SN from TEO is " + str(self.DevSN) + " but expect " + str(self.pa["General"]["TEOExpResp"])
            UtilsCollection.WriteLog(None, [Msg], Pub.CRASHINGMSG)

        ##  ==== HF section: interrogating the maximum gain is a good choice to figure which board we have =========================================
        TEOsmaxGain = self.HF_Gain.GetMaxValue()
        if TEOsmaxGain in [10, 196]:      self.PROBECARDREV = 2
        elif TEOsmaxGain == 26:           self.PROBECARDREV = 1
        else: raise ValueError("as deduced from the gain, the probe card revision is not known")

        if float(self.pa["Aux"]["TEO"]["TEOGain0"]) > 26:
            UtilsCollection.WriteLog(None, ["Input error: check Auxfile, requested TEO gain > 26 and too high"], Pub.CRASHINGMSG)
        self.HF_Gain.SetValue(self.HFgain)
        if not self.HF_Measurement.GetHF_Supported():
            Msg = "Error in TEObox: HF is not reported as supported"
            UtilsCollection.WriteLog(None, [Msg], Pub.CRASHINGMSG)
        self.HF_Measurement.SetHF_Mode(0, not self.TEOintHF)                                                                                        # boolean is whether or not it is external

        # collect items that are needed for switching and internal LFmode
        if not self. LF_Measurement.GetLF_Supported(): UtilsCollection.WriteLog(None, ["LF_Measurement: LF mode not supported"], Pub.CRASHINGMSG)
        self.LF_Measurement.SetLF_Mode(0, not self.TEOintLF)                                                                                        # LFmode is only 0
        self.LFMode = self.LF_Measurement.GetLF_Mode()
        self.LF_Voltage = self.LF_Measurement.LF_Voltage
        self.LFVolt = self.LF_Voltage.GetValue()

        if self.LFMode != 0: raise ValueError("expect TEO's internal LF mode to be 0 for now.")
        self.LF_Voltage.SetValue(0)

        if SHOWINFO:
            print ("TEO: Name/SN/Rev=" + self.DevName + " SN=" + str(self.DevSN) + " Rev=" + str(self.DevRev[0]) + "." + str(self.DevRev[0]))

    def HFGainGetValues(self):
        """
        :return: Dealing with TEO's gain mess. It is one gain value, but two channels. For some reason, he calls the gain steps 'gain' and they go
        opposite. Stick with gain as opposed to steps and communicate via self.HFgainHiBW which makes the whole thing look like the initial version
        """
        GainStep = self.HF_Gain.GetStep()
        self.HFgainLoBW = 0
        if GainStep < 4:
            self.HFgainLoBW = 18 - 6 * self.HF_Gain.GetStep()
        self.HFgain = 26 - GainStep

    def LFCurrent(self):
        """
        :return:
        preliminary..
        TODO need to add safegurad for time, optimize DISCARD, optimize Offset measurement with timestamp
        """

        self.SwitchTo("IV")
        Vs0 = (np.linspace(self.LFVmin, self.LFVmax, self.LFNpts + 1)).tolist()
        Vs = np.zeros(self.LFNpts + 1,)
        Curs = np.zeros(self.LFNpts + 1, )
        t0 = time.time()
        self.GetLFOffset()
        t1 = time.time()

        for i, v in enumerate(Vs0):
            self.LF_Voltage.SetValue(v)
            Vs[i] = self.LF_Voltage.GetValue()
            CurObj = self.LF_Measurement.LF_MeasureCurrent(self.LFtime)
            Cur = CurObj.GetWaveformDataArray()
            Curs[i] = np.mean(Cur[self.DISCARD:]) - self.LFOffset

        t2 = time.time()
        m, b = UtilsCollection.LinearRegression(Vs, Curs)

        plt.plot(Vs, Curs, 'ro')
        plt.plot(Vs, m * Vs + b, 'k--')
        plt.annotate("TestR = 10GΩ", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=8)
        plt.annotate("measured R" + str(1/m/1e9) + "GΩ", xy=(0.05, 0.90), xycoords='axes fraction', fontsize=8)
        plt.annotate("time to get offset " + str(t1-t0), xy=(0.05, 0.85), xycoords='axes fraction', fontsize=8)
        plt.annotate("time to get data " + str(t2-t1), xy=(0.05, 0.8), xycoords='axes fraction', fontsize=8)
        plt.annotate("time to TEO " + str(self.LFtime), xy=(0.05, 0.75), xycoords='axes fraction', fontsize=8)
        Sig = np.std(Curs - m * Vs - b)
        plt.title("fast dc: sigma=" + str(Sig))
        plt.show()

    def MeasureLk(self, Vstart, Vstop, Vstep, Bidirectional=1, CalTEO=True):
        """
        :param          Vstart: voltage to start
        :param          Vstop: voltage to stop
        :param          Vstep: voltage to step
        :param          Bidirectional sweeps up (= to higher voltage) and then down
        :param          can skip TEO's calibration if needed
        :return:  Vs, Imeas, Comment
        Measures current when ramping the voltage up.
        1) COmpatibility: return for Vs and Imeas a list of 1 or 2 lists with the actual numbers
        2) there is an unknown offset.
        """
        t0, MEASTIME, INTERNALPLOT = None, 0.1, False
        if INTERNALPLOT: t0 = time.time()
        _, PtsEVal = GetPoints4TEOsDcMeasurement(Meastime=MEASTIME)
        nPts = (Vstop - Vstart) / Vstep + 1
        if Vstart != 0: raise ValueError("Leakage with internal LF of TEO requires 0 to be the starting voltage")
        Vs = [None]                                                                                                                                 # Pycharm thing
        Vs[0] = np.linspace(Vstart, Vstop, nPts).tolist()

        if Bidirectional == 1:
            Vs.append(np.linspace(Vstop - Vstep / 2, Vstart + Vstep / 2, nPts - 1).tolist())
            Imeas = [[], []]
        else:
            Imeas = [[]]

        self.SwitchTo("IV")
        if CalTEO:
            if self.TEOSwRev == 7075:   raise ValueError("do NOT use version 7075 - messes with RF output")
            elif self.TEOSwRev < 7075:  pass
            else:
                self.LF_Measurement.RunLF_Calibration()
        self.LF_Measurement.SetLF_Mode(0)
        Cal, Off = self.pa["Aux"]["TEO"]["DcCal"], self.pa["Aux"]["TEO"]["DcOff"]                                                                   # see Calibrations: need to do Cal * Ireported + Off
        for ii in range(Bidirectional + 1):
            for vv in Vs[ii]:
                self.LF_Voltage.SetValue(vv)
                dcWF = self.LF_Measurement.LF_MeasureCurrent(MEASTIME)
                Cur = dcWF.GetWaveformDataArray()[-PtsEVal:]
                Imeas[ii].append(np.mean(Cur) * Cal + Off)

        self.LF_Voltage.SetValue(0)
        idxZero = Vs[0].index(0)
        Zero = Imeas[0][idxZero]
        Imeas[0] = [i - Zero for i in Imeas[0]]
        if len(Imeas) > 1: Imeas[1] = [i - Zero for i in Imeas[1]]

        if INTERNALPLOT:
            t1 = time.time() - t0
            I1 = np.asarray(Imeas[0])
            V1 = np.asarray(Vs[0])
            m, b = UtilsCollection.LinearRegression(V1, I1)
            sig = np.std(I1 - m * V1 - b)
            if Bidirectional == 1:
                I1 = np.concatenate((I1, np.asarray(Imeas[1])))
                V1 = np.concatenate((V1, np.asarray(Vs[1])))
            I1 = I1 - Zero
            plt.plot(V1, I1 * 1e12, 'or-')
            plt.annotate("R = " + str(1e-9/m) + "GΩ", xy=(0.01, 0.95), xycoords="axes fraction", fontsize=10)
            plt.annotate("Cur-noise " + str(1e12*sig) + " pA", xy=(0.01, 0.89), xycoords="axes fraction", fontsize=10)
            plt.annotate("WF time " + str(MEASTIME) + " s", xy=(0.01, 0.83), xycoords="axes fraction", fontsize=10)
            plt.annotate("meas. time " + str(t1) + " s", xy=(0.01, 0.77), xycoords="axes fraction", fontsize=10)
            plt.xlabel("V (V)")
            plt.xlabel("I (pA)")
            plt.show()
        Co = ""

        return Vs, Imeas, Co

    def MeasureWFs(self, **Instruct):
        """
        ** Instruct
        ApplyCal:       calibrate waveform
        **SimpleRead:     return raw data without calibration, no syncing. Use for calibration
        **App2RecID:      optional for fOM apply letter to RecID such that we get a family of RecID's
        :return: d - 3D array. 1st index times, 2nd index = 0: voltages, 1 = currents calibrated from file, 2 = Cycle
        it is DIFFERENT for SimpleRead: x, y => voltages and currents (because it would be silly to do all these conversions back and forth)
        This applies to internal TEO only.
        """
        ApplyCal, SimpleRead, Pol, EL, App2RecID = True, False, None, None, ""
        WFinfo = None
        if "ApplyCal" in Instruct:      ApplyCal = Instruct["ApplyCal"]
        if "SimpleRead" in Instruct:    SimpleRead = Instruct["SimpleRead"]
        if "WFinfo" in Instruct:        WFinfo = Instruct["WFinfo"]
        if "App2RecID" in Instruct:     App2RecID = Instruct["App2RecID"]
        d, Status, MONITORTEO = None, 0, False

        VMon = self.AWG_WaveformManager.GetLastResult(0)
        Current = self.AWG_WaveformManager.GetLastResult(1)

        Oload = [VMon.IsSaturated(), Current.IsSaturated()]

        nPts = self.ATLen2Read * self.NCycles
        wflenV = VMon.GetWaveformLength() - 2                                                                                                       # Python for some reason: arrays are reported LONGER by 2
        wflenI = Current.GetWaveformLength() - 2
        if wflenI != wflenV or wflenV != nPts:
            raise ValueError("TEObox: MeasureWFs: arrays do not have the correct length??")

        d = np.zeros((self.ATLen2Read, 2, self.NCycles))
        x = np.asarray(VMon.GetWaveformDataArray()[:nPts])
        y = np.asarray(Current.GetWaveformDataArray())[:nPts]
        if SimpleRead: return x, y
        x = x.reshape((self.NCycles, self.ATLen2Read)).T
        y = y.reshape((self.NCycles, self.ATLen2Read)).T

        TruncL = int(1.2 * WFGeneration.SYNCLEN)

        DoAlign, Trunc, xMax = True, np.ones((self.NCycles,)) * self.ATLen2Read, None
        if ApplyCal:
            EL = 2 * WFGeneration.SYNCLEN + TEOCHUNK                                                                                                # by construction sync pulse is in here
            idx = np.where(np.abs(self.AT) > 0)[0]                                                                                                  # will also work for fOM
            if len(idx) > 0 and idx[0] > self.TEOShift + self.TEOZEROMARG: idx = idx[0] - (self.TEOShift + self.TEOZEROMARG)
            elif np.all(self.AT):
                idx = self.ATLen2Read - 1                                                                                                           # then it is a zero-correction WF
                DoAlign = False
            else: raise ValueError("TEObox-MeasureWFs: cannot get index - why?")
            x = x - self.VMonOffSet * self.VMonCal                                                                                                  # generic 0-correction from file
            y = y - self.CurOffSetUse * self.CurCalUse
            Trunc = np.zeros((self.NCycles,), dtype=int)                                                                                            # needed for trunction

            for cc in range(self.NCycles):                                                                                                          # have to do this for each cycle
                if DoAlign: x[:,cc], y[:,cc], Pol, xMax = self.AlignbyV(EL, x[:,cc], y[:,cc])
                x0 = np.mean(x[:idx])
                y0 = np.mean(y[:idx])
                d[:, 0, cc] = (x[:,cc] - x0) * self.VMonCal                                                                                         # local 0-correction per pulse train
                d[:, 1, cc] = (y[:,cc] - y0) * self.CurCalUse
                if DoAlign: Trunc[cc] = np.where(Pol * d[:, 0, cc] > xMax * self.VMonCal / 2 )[0][-1]

        if MONITORTEO:
            nCy = 0                                                                                                                                 # show 1st cycle
            t = np.asarray(range(self.ATLen2Read))
            Sc = np.max(np.abs(self.AT)) / np.max(np.max(d[:,1, nCy - 1]))
            plt.subplots(2, 1)
            plt.subplot(2,1,1)
            plt.plot(t, self.AT, "0.5")
            plt.plot(t, d[:, 0, nCy], "g")
            plt.plot(t, d[:, 1, nCy] * Sc * 0.9, "r")
            plt.xlabel("sample")
            plt.ylabel("V")
            plt.subplot(2,1,2)
            plt.plot(np.asarray(range(EL)), self.AT[-EL:] / Sc * 0.9, "0.5")
            plt.plot(np.asarray(range(EL)), d[-EL:, 0, nCy] / Sc * 0.9 * 1000, "g")
            plt.plot(np.asarray(range(EL)), d[-EL:, 1, nCy] * 1000, "r")
            plt.xlabel("sample")
            plt.ylabel("I (mA)")

            # noise est
            idx0 = np.where(np.abs(self.AT) > 0)[0]
            if len(idx0) > 10: idx0 = idx0[0] - 10
            Noi = np.std(d[:idx0, 1, nCy - 1])
            Leg = ["Noise/Pt " + "{0:.3f}".format(Noi * 1000000) + "µA", "Cal V-Mon " + "{0:.3f}".format(self.VMonCal),
                   "Cal-I " + "{0:.6f}".format(self.CurCalUse) + " A/V"]
            for ii, LL in enumerate(Leg):
                plt.annotate(LL, xy=(0.01, 1 - ((ii + 1) * 0.06)), xycoords="axes fraction", fontsize=7)
            plt.show()

        Trunc = int(np.mean(Trunc)) - TruncL
        d = d[:Trunc,:,:]
        Status = self.StoreTraces(d, App2RecID=App2RecID)
        Status["OverFlow"] = Oload
        return d, Status

    def PadWaveform(self, AT, MakeLikeRigol=False):
        """
        :param AT:                  waveform as generated
        :param MakeLikeRigol:       pad the WF in the same way as we do it for the Rigol.
        :return: At = padded waveform
        Normally it pads the WF's such that their length is a multiple of 2048. Actually not needed, bu good to have a bit of a gap between cycles
        If the Rigol flag is true, it pads in the same way as we do it for the Rigol. This uses its magic numbers. May be needed occasionally
        """
        Chunk = TEOCHUNK
        if MakeLikeRigol:
            Chunk = self.MAGICLENGTHS[-1]
            if len(AT) < self.MAGICLENGTHS[-1]: Chunk = next(x for x in self.MAGICLENGTHS if x > len(AT))
        nChunk = int(np.ceil(len(AT) / Chunk))
        pad = 0
        if len(AT) < nChunk * Chunk: pad = nChunk * Chunk - len(AT)
        if pad > 0: AT = np.concatenate((AT, np.zeros(pad, )))
        return AT, pad

    def ProgramAWG(self, A, Ampl, WFname, **Instruct):#A=None, Ampl=None, WFname="test", ForceProgram=True, Gate=None):
        """
        :param A:                   actual sequence, normalized
        :param Ampl:                maximum amplitude - remember Cal
        :param WFname:              name of WF, can still be "" which then creates test WF
        ** various instructions, see below
        :return:
        These are things similar as we produce them with the Rigol (the end is shorter). If A is None, it generates a test waveform
        Before programming, it checks using a WFId whether the WF has been programmed and whether it is identical. One could ask the TEO board, but
        that takes longer because the whole WF needs to be transmitted. Just the length does not suffice. Thi sis faster than reprogramming
        IMPORTANT: similar to the PickOffT, need to remove the calibration factor between the power output and the input. Therefore, tis routine has
        to be called by Ampl/Cal. To keep consistence, the 50Ohm termination needs to be treated in the same way as for the Rigol and since the cal
        factor contains this, the factor 2 appears in self.AT = self.AT * Ampl * 2. If it is indeed termianted with 50 Ohm, Cal is close to 1.
        """
        if not os.path.isdir(Pub.PATH4WFIDS): raise ValueError("You need to create a folder 'WfIDs' in the Setups folder")

        ForceProgram, Gate, Trig2ForThings = True, None, None

        if "ForceProgram" in Instruct:                  ForceProgram = Instruct["ForceProgram"]                                                     # force programming, even if WF is there
        if "Gate" in Instruct:                          Gate = Instruct["Gate"]                                                                     # applies to Trig1: sets the gate such that inter TEO reads inside
        if "Trig2ForThings" in Instruct:                Trig2ForThings = Instruct["Trig2ForThings"]                                                 # applies to Trig1: doe snot affect TEO, but create trigger signals for ext instruments
        if Gate is not None and Trig2ForThings is not None:
            raise ValueError("ProgramAWG does not simulateously support internal reads with gate and external gates")

        if WFname == "":                                                                                                                            # still have loophole for test WF
            self.AT, Ampl = GenerateTestWaveform()
            WFname = "test"
        else:
            self.AT = np.asarray(A, dtype=float)

        self.AT = self.AT * Ampl * 2

        self.AT, pad = self.PadWaveform(self.AT)
        self.ATLen = self.AT.size
        self.TEOProgDur = self.TEOTS * self.ATLen

        # with open("C:\\Users\\User\\Desktop\\TEO dev\\Glitches\\WF1.csv", 'w', newline="") as f:
        #     writer = csv.writer(f, delimiter=",", escapechar=" ", quoting=csv.QUOTE_NONE)
        #     for Bd in self.AT.tolist():
        #         writer.writerow([Bd])
        self.ATLen2Read = self.AT.size
        if Gate is not None:
            self.ATLen2Read = np.count_nonzero(Gate) + pad
        # =============================== check whether WF has been programmed. If so and they are the same, skip.
        WFwanted = {"Length": len(self.AT), "Sum": sum(self.AT), "Sigma": np.std(self.AT), "Dt": self.TEOTS, "PrgLength": self.TEOProgDur,
                    "Dff": np.sum(np.diff(self.AT)), "LenRec": self.ATLen2Read}                                                                     # similar to Rigol need Dff to distinguish similar WFs
        WFnameP = Pub.PATH4WFIDS + "\\" + WFname + ".JSON"

        HaveWF = os.path.isfile(WFnameP)
        if HaveWF:
            WFonFile = UtilsCollection.ReadDictJson(WFnameP)
            HaveWF = WFwanted == WFonFile

        if HaveWF:
            wfN = -1
            WFsinMemory = []
            Search = True
            while Search:
                wfN = wfN + 1
                WFmem = self.AWG_WaveformManager.GetWaveformName(wfN)
                Search = WFmem != ""
                if WFmem !="": WFsinMemory.append(WFmem)
            HaveWF = WFname in WFsinMemory

        if HaveWF and not ForceProgram: return WFnameP
        if HaveWF: self.AWG_WaveformManager.DeleteWaveform(WFname)
        self.wf = self.AWG_WaveformManager.CreateWaveform(WFname)
        Trig1 = np.full((self.ATLen,), True, dtype=bool)

        if Trig2ForThings is not None:                                                                                                              # Gate and this are mutually exclusive
            Trig1 = np.concatenate((Trig1, np.full((pad,), True, dtype=bool)))
            Trig2 = np.concatenate((Trig2ForThings, np.full((pad,), True, dtype=bool)))
            self.wf.AddSamples(self.AT, Trig1, Trig2)
        else:
            if Gate is not None:
                Trig1 = np.concatenate((Gate, np.full((pad,), True, dtype=bool)))                                                                   # need to get the padding, must record that part
            self.wf.AddSamples(self.AT, Trig1, Trig1)                                                                                               # categorically set the 2nd the same as the first

        print("Creating waveform " + WFname + "  " + str(self.ATLen)+ " sending " + str(self.ATLen) + " " + str(Ampl * 2) + "V")
        UtilsCollection.WriteDictJson(WFnameP, WFwanted)
        self.WFGlitchFree, self.LastProgedWFName = True, WFname
        return WFname

    def ProgramAWGgeneral(self, tIN, A, Ampl, Cal, WFname):
        """
        :param tIN:         time sequence for AWG (ignore for TEO)
        :param A:           amplitude sequence, normalized to [-1, 1]
        :param Ampl:        max amplitude (incl OS and all that).
        :param Cal:         calibration factor to undo calibration.
        :param WFname:      name of waveform. Ignore for Rigol - there is only one.
        :return:            ProgDur = actual time that was programmed. report that back because the picoscope needs it.
        """
        # careful. Because of the binary programming, the programmed WF can be longer than the intended one. If there is one cycle, this does not
        # matter because the rest of the WF are zeros anyway and there is no need to record these. However, if there is more than one cycle, the
        # picoscope will not catch the entire WF. The evaluation automatically finds the end of the WF.

        if self.HalfTEO:
            self.ProgramAWG(A, Ampl / Cal, WFname, ForceProgram=False)
            ProgDur = self.TEOProgDur
        else:
            # this undoes the calibration. Example: the AWG puts out 5 V if 4 are requested (assuming cal = 0.8)
            self.Aw.WriteWF2AWG(tIN, A, Ampl / Cal)
            ProgDur = self.Aw.WFProgrDur
        return ProgDur

    def SelectCalibration(self, GaindB, IsCalibration=False):
        """
        :param GaindB:            refers to gain (NOT steps!) for the HiBW amplifier as in the initial version.
        :param IsCalibration:     refers to gain (NOT steps!) for the HiBW amplifier as in the initial version.
        :return:
        It selects the appropriate offset and gain for the selected gain and reports this into the correct variables. This is rigorous for the low
        bandwidth channel and internal operation. The smallest index (0) is the highest gain.
        The gains of external channels are calculated and their offsets are not recorded.
        For the high BW channel, the calibration refers to the gain that was used at calibration (traditional).
        """
        if not self.FullTEO: return
        if GaindB == -99: return
        idxLo = 3                                                                                                                                   # order :0 = highest, then down

        self.LineDly = int(self.LineDly_ns / DELTAT)                                                                                                # set all current delays. Relative to voltage.
        self.LineDlyX = int(self.LineDlyX_ns / DELTAT)
        self.LineDlyXFull = int(self.LineDlyXFull_ns / DELTAT)

        if GaindB > 22: idxLo = int(26 - GaindB)
        GainLo0 = max([self.pa["Aux"]["TEO"]["TEOGain0"], 23])
        GainLo = max([GaindB, 23])
        self.HFgainLoBW = GainLo0 + 6 * (GainLo - GainLo0)
        GainCorrX = 10**(-(6 * (GainLo - GainLo0)) / 20)
        GainCorrFullX = 10 ** ((GaindB - self.pa["Aux"]["TEO"]["TEOGain0"]) / 20)

        if IsCalibration:
            self.CurOffSetUse = 0
            self.CurCalUse = 1
            self.CurCalX = 1
            self.CurCalFullX = 1
            self.VMonOffSet = 0
            self.VMonCal = 1
            self.pa["Aux"]["PickOffTCal"] = 1
        else:
            if self.HiSenseR:
                self.CurOffSetUse = self.CurOffsetHiR[idxLo]
                self.CurCalUse = self.CurCalHiR[idxLo]
                self.CurCalX = self.pa["Aux"]["TEO"]["CurCalHiRX"] * GainCorrX
                self.CurCalFullX = self.pa["Aux"]["TEO"]["CurCalHiRX"] * GainCorrFullX
                self.NoisepPtUse = self.NoisepPtHiR[idxLo]
                self.SaturationUse = [11, self.pa["Aux"]["TEO"]["SatHiR"][idxLo]]                                                                   # meant for integer conversion. 11 has built-in margin for voltage
            else:
                self.CurOffSetUse = self.CurOffsetLoR[idxLo]
                self.CurCalUse = self.CurCalLoR[idxLo]
                self.NoisepPtUse = self.NoisepPtLoR[idxLo]
                self.SaturationUse = [11, self.pa["Aux"]["TEO"]["SatLoR"][idxLo]]
                self.CurCalX = self.pa["Aux"]["TEO"]["CurCalLoRX"] * GainCorrX
                self.CurCalFullX = self.pa["Aux"]["TEO"]["CurCalLoRX"] * GainCorrFullX
            self.VMonOffSet = self.pa["Aux"]["TEO"]["VMonOffset"]
            self.VMonCal = self.pa["Aux"]["TEO"]["VMonCal"]
        self.SelectedGain = GaindB
        return

    def SetAWGCycles(self, NCycl):
        """
        :param NCycl:          # of cycles to use
        :return:
        Wrapper to be called from outside (MeasurementsCore). It accommodates using "half TEO operation". only the TEOmodule knows what it is.
        Selecting between TEO's AWG + picosope and Rigol + picoscope
        """
        if self.HalfTEO or self.FullTEO:
            self.NCycles = NCycl
        else:                                                                                                                                       # rigol
            self.Aw.pa["NCycl"] = NCycl

    def SetHFGain(self, NewGaindB):
        """
        :param          NewGaindB:  refers to gain (NOT steps!) for the HiBW amplifier as in the initial version.
        :return:
        """
        StepVal = 26 - NewGaindB
        self.HF_Gain.SetStep(StepVal)
        self.HFGainGetValues()                                                                                                                      # update gain values

    def StoreTraces(self, d, App2RecID=""):
        """
        :param d:           Data array (calibrated). Time / 0 = V, 1 = I / cycle

        :return:   Status dict
        store data in float and and meta data as usual
        """
        STOREASINT, CHECKSTORAGE = True, False
        Status = {"RecID": self.RecID}

        Npts, Nch, NCycles = d.shape
        if NCycles != self.NCycles: raise ValueError("StoreTraces in TEOModule: # of cycles is wrong")
        Status["NCycl"] = self.NCycles  # for reshaping data
        Status["Npts"] = [Npts]                                                                                                                     # this signals that we are use TEO internal
        Status["OverFlow"] = [False] * 2

        Status["Ts"] = self.TEOTS
        # # Careful: need to keep the range and the calibration separate, because the offset from the picoscope is a voltage
        Status["TEOgain"] = self.HFgain                                                                                                             # main gain - FYI
        Status["TEOgainLoBW"] = self.HFgainLoBW                                                                                                     # gain - FYI
        Status["Cal"] = [self.VMonCal, self.CurCalUse]                                                                                              # FYI

        Status["ySteps2"] = 2**11                                                                                                                   # 12 bit
        Status["LineDly_ns"] = self.LineDly_ns                                                                                                      # FYI
        if STOREASINT:
            Status["IntConv"] = [32768 / (self.SaturationUse[0] / self.VMonCal), 30000 / self.SaturationUse[1]]                                     # map this into the 16bit integere range of -16384 to 16384
        else:
            Status["IntConv"] = [1, 1]

        """ Store the trace file as a binary file with ending 'bin' and the parameters as a dict in a json file"""
        if self.RecID == "":
            self.TrcFolder = self.pa["General"]["DataDrive"] + "\\Test"
            if not os.path.isdir(self.TrcFolder): os.mkdir(self.TrcFolder)
            JsonFile = self.TrcFolder + "\\test.JSON"
            BinFile = self.TrcFolder + "\\test.bin"
        else:
            if App2RecID == "":
                JsonFile = self.RecIDwPathJSON
                BinFile = self.RecIDwPathBIN
            else:
                JsonFile = self.RecIDwPathJSON[:-5] + App2RecID + ".JSON"
                BinFile = self.RecIDwPathBIN[:-4] + App2RecID + ".bin"
        with open(JsonFile, 'w') as fp:
            json.dump(Status, fp, indent=2)

        if STOREASINT:
            ds = np.zeros(d.shape, dtype="int16")
            ds[:, 0, :] = np.round(d[:, 0, :] * Status["IntConv"][0])                                                                                            # voltages.
            ds[:, 1, :] = np.round(d[:, 1, :] * Status["IntConv"][1])                                                                                            # currents
            (ds.astype("int16")).tofile(BinFile)
        else:
            d.tofile(BinFile)

        if not CHECKSTORAGE: return Status
        # use this to test ...

        _, x = AnaBits.LoadTraceFile(BinFile[:-4], paTrc=Status)
        for cc in range(NCycles):
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.plot(d[:, 0, cc - 1], 'g')
            ax1.plot(1000*(x[:, 0, cc - 1] - d[:, 0, cc - 1]), 'c')
            ax2.plot(d[:, 1, cc - 1], 'r')
            ax2.plot(1000*(x[:, 1, cc - 1] - d[:, 1, cc - 1]), 'm')
        plt.show()
        return Status

    def SwitchTo(self, State, Ask=True):
        """
        :param State:           State to switch to
        :param Ask:             asks for cables for calibration for FullTeo
        :return:
        Contact to the outside world for switching. Here it forks between TEO and the switchbox
        """
        SkipThis = False
        if self.TEOactive:
            if State == "Pulsing":
                self.HF_Measurement.SetHF_Mode(0, self.TEOextHF)
            elif State == "CalI":
                print("**********************************************")
                print("Calibration for TEO's output amplifiers. Mode = mixed with PicoScope, rev 1 board")
                print("Connect the Keithley 2400 with the little box to TEO's current (measurement) input. ")
                print("Connect the Rigol via a " + str(self.TEOattndB) + " dB ATTENUATOR to the AWG input J6")
                print("Connect the picoscope channels A(blue)/B(red)/D(yellow) to J10/J11/J26")
                print("Connect the picoscope channel C (green) the monitor output J9")
                print("NOTE: if the voltage channel goes into overload: C:NVMPyMeas\setups\AuxiliaryInfo.json change Rng for channel C")
                Resp = UtilsCollection.MessageBox("is Keithley 2400 connected?", "yn", "Calibration: TEO - I")
                if not Resp: UtilsCollection.WriteLog(None, ["user terminated"], Pub.CRASHING)
                self.HF_Measurement.SetHF_Mode(0, self.TEOextHF)
            elif State == "CalIFullTEO":
                if Ask:
                    print("**********************************************")
                    print("Calibration for TEO's output amplifiers. Mode = fully internal, rev 2 board")
                    print("You should be using the Rev 2 board - if not, it is WRONG.")

                    print("Connect the Keithley 2400 with the little box to TEO's current (measurement) input. ")
                    print("PicoScope channel A (blue): connect with 20dB attenuator (IMPORTANT) to main voltage output")
                    print("PicoScope channel B (red): connect to limited BW amplifier output on board")
                    print("PicoScope channel C (green): connect voltage monitor output on board")
                    print("PicoScope channel D (yellow): connect to limited BW amplifier output on board")
                    print("PicoScope external trigger: connect to Trigger1 from TEO board")
                    Resp = UtilsCollection.MessageBox("All cables connected?", "yn", "Calibration: CalFullTEO")
                    if not Resp: UtilsCollection.WriteLog(None, ["user terminated"], Pub.CRASHING)

                self.HF_Measurement.SetHF_Mode(0, self.TEOextHF)
            elif State == "CalV":
                print("**********************************************")
                print("Calibration for TEO's main amplifier. ")
                print("Please connect the GREEN (normally voltage control) channel of the picoscope to TEO's RF amplifier output (Voltage) - J28 = HF-V")
                print("Please connect the output of the Rigol via a " + str(self.TEOattndB) + " dB ATTENUATOR to the AWG input J6")
                Resp = UtilsCollection.MessageBox("Both connected?","ync", "Calibration: TEO - V")
                if Resp is None:
                    SkipThis = True
                else:
                    if not Resp: UtilsCollection.WriteLog(None, ["user terminated"], Pub.CRASHING)

                self.HF_Measurement.SetHF_Mode(0, self.TEOextHF)
            elif State == "IV":
                self.LF_Measurement.SetLF_Mode(self.LFMode, not self.TEOintLF)
                if SHOWINFO: print("TEO-LF: " + self.LF_Measurement.GetLF_ModeDescription(0))
            else:
                UtilsCollection.WriteLog(None, ["wrong mode in switch to in TEObox"], Pub.PROGERR)
        else:
            if not isinstance(self.Sw, InstrSwitch.Switchclass):
                pa2Switch = {"Addr": self.pa["General"]["Switchaddress"], "ExpResp": "norespcheck"}
                self.Sw = InstrSwitch.Switchclass(pa2Switch)
            self.Sw.SwitchTo(State)
        return SkipThis

    def TestLF(self):
        """
        uncontrolled
        :return:
        """
        # self.LF_Measure = self.LF_Measurement.LF_Measure()
        if not self.TEOintLF: raise ValueError("You need to change testsystem to interal LF to do this")
        #self.LFCurrent()
        self.LF_Measurement.RunLF_Calibration() # switches to IV
        self.SwitchTo("IV")
        self.LF_Measurement.SetLF_Mode(0)
        self.LF_Voltage.SetValue(-10)
        print(self.LF_Voltage.GetValue())
        test = self.LF_Measurement.LF_MeasureCurrent(0.1)

        print(" length "  + str(test.GetWaveformLength()))
        print(" sampling rate " + str(test.GetWaveformSamplingRate()))
        print(" waveform name " + test.GetWaveformName())
        plt.plot(test.GetWaveformDataArray())
        plt.show()
        self.LF_Voltage.SetValue(0)
        print("end LF test")

    def TriggerAWGgeneral(self, WFname, TimeIT=False):
        """
        :param WFname:          name of WF, ignore for Rigol
        :param TimeIT:          Full TEO only - reports the estimated time when the WF was initiated
        :return:
        Wrapper to be called from outside (MeasurementsCore). It accommodates using "half TEO operation" - only the TEOmodule knows what AWG to use.
        Picoscope needs some time to get ready. Give it some time dependent on WF, min min 10ms
        """
        if self.HalfTEO:
            PicoThink = 0.01 * self.TEOProgDur / self.TEOTS / 194560                                                                                # calibrated to yield 10ms for test sequence 194560 points
            if PicoThink < 0.01: PicoThink = 0.01
            time.sleep(PicoThink)
            self.AWG_WaveformManager.Run(WFname, int(self.NCycles))
        elif self.FullTEO:
            if TimeIT: self.BeforeTrig = time.time()                                                                                                # clock thing is uncontrollable
            self.AWG_WaveformManager.Run(WFname, int(self.NCycles))
            if TimeIT: self.AfterTrig = time.time()
        else:
            self.Aw.TriggerAWG()                                                                                                                    # Rigol - no name just 1 WF

if __name__ == "__main__":
    paX = UtilsCollection.GetAuxInfoPlus()
    paX["General"] = UtilsCollection.GetGeneralInfo()
    #DeleteWFIDs()
    with TEObox(paX,["AWG"]) as teo:
        #teo.Test()
        teo.TestLF()
        #teo.AWG_WaveformManager.Run("VTh", int(teo.NCycles))
        print(" SN= " + str(teo.DevSN))
