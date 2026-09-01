from __future__ import annotations

import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from uncertainties import ufloat as uf

from sysvar.utils import SavableAttributesObject
from sysvar.visualize import Visualizer, DPI, CMAP, get_latex_symbol

logging.basicConfig(
    format="%(levelname)s :  %(message)s",
    level=logging.INFO,
)


def show_available_models(model: str | None = None):

    logging.warning(
        "If the model parameters you are interested in are not available, you need to implement a classmethod with the the parameters of interest. Take a look at sysvar.ff_models for examples"
    )

    model_found = False
    for subclass in FFModel.__subclasses__():

        if model is not None and model != subclass.__name__:
            continue
        else:
            model_found = True

        for key, attr in subclass.__dict__.items():

            if isinstance(attr, classmethod):
                model = attr.__func__(subclass)

                print_model_info(model, key)

    if not model_found:
        print(
            f"You specified the model of interest as {model}. However no implemented class was found. Make sure the model of interest is one of the {[x.__name__ for x in FFModel.__subclasses__()]}"
        )


def print_model_info(
    model, classmethod_name="", save: bool = False, filename: str = ""
):

    print(40 * "*")
    print("Form Factor model: ", model.name)
    if classmethod_name != "":
        print("classmethod implementation: ", classmethod_name)
    print(model.expert_info)
    print(40 * "*")
    visualizer = FFModelVisualizer(model)
    if model.corr_matrix is not None:
        ax = visualizer.plot_corr_and_params(save=save, filename=filename)
    else:
        ax = visualizer.plot_params(save=save, filename=filename)
    plt.show()


class FFModel(SavableAttributesObject):
    def __init__(
        self,
        name: str,
        Xc: str = None,
        lep: str = None,
        params: dict = None,
        param_init: dict = None,
        corr_matrix: np.ndarray | dict = None,
        expert_info: str = None,
    ):

        super().__init__()
        self.name = name
        self.Xc = Xc
        self.lep = lep
        self.params = params
        self.param_init = param_init
        self.corr_matrix = corr_matrix
        self.expert_info: str = expert_info

    @property
    def num_params(self):
        return len(self.params.keys())

    @property
    def parameter_central_values(self):
        return [
            param.nominal_value if name != "DelMbc" else param["mc"].nominal_value
            for name, param in self.params.items()
        ]

    @property
    def parameter_errors(self):
        return [
            param.std_dev if name != "DelMbc" else param["mc"].std_dev
            for name, param in self.params.items()
        ]


class BGL(FFModel):
    def __init__(
        self,
        Xc: str,
        lep: str,
        params: dict,
        param_init: dict,
        corr_matrix: np.ndarray | dict,
        expert_info: str,
    ):
        super().__init__("BGL", Xc, lep, params, param_init, corr_matrix, expert_info)

    @classmethod
    def BtoDEllNu_dec_outdated(cls):
        """
         # B -> D FFs from Svenja: wg1_b2pilnu/Systematics/D_FF.ipynb
        # Svenja claims that these are the values that Florian calculated himself and
        # they are not published yet.
        # this model is used by most people in Bonn.
        """

        return cls(
            Xc="D",
            lep="Ell",
            params={
                "a+_1": uf(1.261470e-02, 9.788790e-05),
                "a+_2": uf(-9.620840e-02, 3.341480e-03),
                "a+_3": uf(4.138840e-01, 9.444710e-02),
                "a+_4": uf(-1.736990e-01, 8.918090e-01),
            },
            param_init="{ap: [%f,%f,%f,%f]}",
            corr_matrix=np.array(
                [
                    [1.000, 0.245, -0.161, 0.020],
                    [0.245, 1.000, -0.654, 0.272],
                    [-0.161, -0.654, 1.000, -0.770],
                    [0.020, 0.272, -0.770, 1.000],
                ]
            ),
            expert_info=cls.BtoDEllNu_dec_outdated.__func__.__doc__,
        )

    @classmethod
    def BtoDEllNu_dec_updated(cls):
        """
        B -> D BGL with correct Blaschke factors, Schwanda refit of arXiv:1510.03657.
        Parameters match BGLB2.BtoDEllNu_dec_updated() but use standard BGL Blaschke factors.
        Use as the reweighting target (input uses BGLB2 to match EvtGen's wrong Blaschke).
        """
        return cls(
            Xc="D",
            lep="Ell",
            params={
                "a+_1": uf(0.0127, 0.0001),
                "a+_2": uf(-0.095, 0.003),
                "a+_3": uf(0.393, 0.157),
                "a+_4": uf(-0.577, 2.244),
                "a0_1": uf(0.0114, 0.0001),
                "a0_2": uf(-0.058, 0.003),
                "a0_3": uf(0.231, 0.139),
                "a0_4": uf(-0.789, 2.143),
            },
            param_init="{ap: [%f,%f,%f,%f], a0: [%f,%f,%f,%f]}",
            corr_matrix=np.array([
                [1.000, 0.130, -0.028, -0.042, 0.000, 0.122, 0.091, -0.108],
                [0.130, 1.000, -0.575, 0.349, 0.000, 0.720, -0.170, 0.029],
                [-0.028, -0.575, 1.000, -0.929, 0.000, -0.407, 0.649, -0.586],
                [-0.042, 0.349, -0.929, 1.000, 0.000, 0.254, -0.690, 0.750],
                [0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000],
                [0.122, 0.720, -0.407, 0.254, 0.000, 1.000, -0.438, 0.244],
                [0.091, -0.170, 0.649, -0.690, 0.000, -0.438, 1.000, -0.929],
                [-0.108, 0.029, -0.586, 0.750, 0.000, 0.244, -0.929, 1.000],
            ]),
            expert_info=cls.BtoDEllNu_dec_updated.__func__.__doc__,
        )

    @classmethod
    def BtoDEllNu_ins2936544(cls):
        """
        B -> D BGL refit with Hammer's actual Blaschke factors, constrained to
        HEPData ins2936544 (B -> D ell nu dGamma/dw, 10 bins, w in [1.0, 1.59])
        plus MILC+HPQCD lattice QCD inputs. Fit via DlDetal/.../sem-hausdorff
        investigations/ff_impl_crosscheck/fit_bgl_hammer_params.py, chi2/dof =
        8.34/13. a0_1 is derived from the f+(wmax)=f0(wmax) kinematic constraint;
        its uncertainty/correlations are propagated from the other 7 fitted
        parameters (not treated as independent). Vcb = 39.53 +/- 0.67 (x1e-3)
        from the fit but not used here (cancels in the ff_weight ratio).
        Supersedes BtoDEllNu_dec_updated() (Schwanda refit of arXiv:1510.03657)
        as the reweighting target — re-run the fit script if newer HEPData/lattice
        inputs become available.
        """
        return cls(
            Xc="D",
            lep="Ell",
            params={
                "a+_1": uf(0.015545, 0.000107),
                "a+_2": uf(-0.034611, 0.003211),
                "a+_3": uf(-0.129226, 0.089293),
                "a+_4": uf(1.411815, 2.634988),
                "a0_1": uf(0.077607, 0.000531),
                "a0_2": uf(-0.234187, 0.020609),
                "a0_3": uf(2.118891, 1.462642),
                "a0_4": uf(-24.649408, 22.059403),
            },
            param_init="{ap: [%f,%f,%f,%f], a0: [%f,%f,%f,%f]}",
            corr_matrix=np.array([
                [1.00000000, 0.19721669, 0.05409357, -0.08631800, 0.78252090, 0.19650267, -0.03435580, 0.00950549],
                [0.19721669, 1.00000000, -0.02196083, -0.12256575, 0.22697624, 0.25302663, 0.36942355, -0.34607050],
                [0.05409357, -0.02196083, 1.00000000, -0.93130260, 0.06648694, 0.15849086, 0.19042852, -0.47480010],
                [-0.08631800, -0.12256575, -0.93130260, 1.00000000, -0.09573941, -0.07097244, -0.30492472, 0.60928165],
                [0.78252090, 0.22697624, 0.06648694, -0.09573941, 1.00000000, 0.03927568, -0.03804134, 0.01325128],
                [0.19650267, 0.25302663, 0.15849086, -0.07097244, 0.03927568, 1.00000000, -0.65329344, 0.51143167],
                [-0.03435580, 0.36942355, 0.19042852, -0.30492472, -0.03804134, -0.65329344, 1.00000000, -0.93738500],
                [0.00950549, -0.34607050, -0.47480010, 0.60928165, 0.01325128, 0.51143167, -0.93738500, 1.00000000],
            ]),
            expert_info=cls.BtoDEllNu_ins2936544.__func__.__doc__,
        )

    @classmethod
    def BtoDstEllNu_Scaled(cls, tau=False):
        """
        B -> D* FFs from Svenja (wg1_b2pilnu/Systematics/B2DstFF.ipynb), BGL(1,1,2) parameters of
        https://arxiv.org/pdf/2008.09341v1.pdf table VI, as used (already Vcb-scaled) in the dec file:
        https://stash.desy.de/projects/B2/repos/software/browse/decfiles/dec/DECAY_BELLE2.DEC?at=refs%2Ftags%2Frelease-05-02-00#345,369,3873,3900
        Only used for central values, so no correlations/errors needed; Vcb is 1.0/1.0066 but irrelevant here.
        EvtGen's BGL sets a0f=0 (equivalent to Hammer's d-vector=0), cf.
        https://stash.desy.de/projects/B2/repos/basf2/browse/generators/evtgen/models/src/EvtBGLFF.cc#141 vs
        https://gitlab.com/mpapucci/Hammer/-/blob/development/src/FormFactors/BGL/FFBtoDstarBGL.cc#L175
        Note: Hammer's own default parameters (https://arxiv.org/pdf/1902.09553.pdf) differ substantially from
        these Belle II inputs, for reasons not fully understood (possibly differing lattice QCD treatment or
        parameter definitions) - the shapes were cross-checked and agree in https://arxiv.org/pdf/1908.09398.pdf fig. 2.
        """

        return cls(
            Xc="D*",
            lep="Ell" if not tau else "Tau",
            params={
                "ag_0": uf(0.02596, 0),
                "ag_1": uf(-0.06049, 0),
                "af_0": uf(0.01311, 0),
                "af_1": uf(0.01713, 0),
                "aF1_1": uf(0.00753, 0),
                "aF1_2": uf(-0.09346, 0),
            },
            param_init="{avec: [%f,%f], bvec: [%f,%f], cvec: [%f,%f], dvec: [0.0, 0.0], Vcb: 0.993443}",
            corr_matrix=np.identity(6),
            expert_info=cls.BtoDstEllNu_Scaled.__func__.__doc__,
        )

    @classmethod
    def BtoDstEllNu_dec(cls):
        """
        Vcb is 34.9/(0.906*1.0066) - the following are the unscaled parameters that need to serve as the input for Hammer:
        """

        return cls(
            Xc="D*",
            lep="Ell",
            params={
                "ag_0": uf(1.00e-03, 0),
                "ag_1": uf(-2.33e-03, 0),
                "af_0": uf(5.05e-04, 0),
                "af_1": uf(6.60e-04, 0),
                "aF1_1": uf(2.90e-04, 0),
                "aF1_2": uf(-3.60e-03, 0),
            },
            param_init="{avec: [%f,%f], bvec: [%f,%f], cvec: [%f,%f], dvec: [0.0, 0.0], Vcb: 0.0382684}",
            corr_matrix=None,
            expert_info=cls.BtoDstEllNu_dec.__func__.__doc__,
        )

    @classmethod
    def BtoDstEllNu_B2dec(cls):
        """
        The following are the unscaled parameters that need to serve as the input for Hammer:
        Parameters taken from https://arxiv.org/pdf/2008.09341.pdf Table V p.9.
        Neet to confirm that BGL(1,1,2) is the correct one.
        """

        return cls(
            Xc="D*",
            lep="Ell",
            params={
                "ag_0": uf(1.00e-03, 0),
                "ag_1": uf(-2.35e-03, 0),
                "af_0": uf(5.11e-04, 0),
                "af_1": uf(6.70e-04, 0),
                "aF1_1": uf(3.00e-04, 0),
                "aF1_2": uf(-3.68e-03, 0),
            },
            param_init="{avec: [%f,%f], bvec: [%f,%f], cvec: [%f,%f], dvec: [0.0, 0.0], Vcb: 0.0382684}",
            corr_matrix=None,
            expert_info=cls.BtoDstEllNu_B2dec.__func__.__doc__,
        )

    @classmethod
    def BtoDstEllNu_new(cls):
        """
        B -> D* BGL fit, Table VII. BGL(2,2,1): 3 coefficients each for avec (g FF) and bvec
        (f FF), 2 for cvec (F1 FF). |Vcb| = 40.97e-3 from the fit (hardcoded in param_init;
        cancels in ff_weight ratio). Correlation matrix is 8x8 (Vcb row/column excluded).
        """

        return cls(
            Xc="D*",
            lep="Ell",
            params={
                "ag_0":  uf( 0.0268,  0.0006),
                "ag_1":  uf(-0.0265,  0.0235),
                "ag_2":  uf(-1.4518,  0.8735),
                "af_0":  uf( 0.0130,  0.0001),
                "af_1":  uf( 0.0130,  0.0066),
                "af_2":  uf(-0.1150,  0.2240),
                "aF1_1": uf(-0.0025,  0.0016),
                "aF1_2": uf( 0.0078,  0.0329),
            },
            param_init="{avec: [%f,%f,%f], bvec: [%f,%f,%f], cvec: [%f,%f], dvec: [0.0, 0.0], Vcb: 0.04097}",
            corr_matrix=np.array([
                [ 1.00,  0.03, -0.11,  0.17,  0.00, -0.02,  0.08, -0.04],
                [ 0.03,  1.00, -0.67, -0.01,  0.17, -0.03,  0.09, -0.03],
                [-0.11, -0.67,  1.00,  0.03, -0.02, -0.03,  0.01, -0.05],
                [ 0.17, -0.01,  0.03,  1.00, -0.02,  0.02,  0.02, -0.03],
                [ 0.00,  0.17, -0.02, -0.02,  1.00, -0.66,  0.59, -0.46],
                [-0.02, -0.03, -0.03,  0.02, -0.66,  1.00, -0.45,  0.40],
                [ 0.08,  0.09,  0.01,  0.02,  0.59, -0.45,  1.00, -0.82],
                [-0.04, -0.03, -0.05, -0.03, -0.46,  0.40, -0.82,  1.00],
            ]),
            expert_info=cls.BtoDstEllNu_new.__func__.__doc__,
        )


class BGLB2(FFModel):
    def __init__(
        self,
        Xc: str,
        lep: str,
        params: dict,
        param_init: dict,
        corr_matrix: np.ndarray | dict,
        expert_info: str,
    ):
        super().__init__(
            "BGLB2", Xc, lep, params, param_init, corr_matrix, expert_info
        )  # wrong Belle II BGL implementation -> extra class in Hammer needed!

    @classmethod
    def BtoDEllNu_dec(cls):
        """
        B ->D FFs from Stephan.
        https://stash.desy.de/projects/B2A/repos/sduell_phaseii/browse/scripts/pyHammerModule.py#458-474.
        These are four weights for B+ and four for B0 and the same correlations are assumed.
        https://arxiv.org/pdf/1510.03657.pdf page 14 for N = 3, N being the number of paramters.
        Also see the dec file:
        https://stash.desy.de/projects/B2/repos/software/browse/decfiles/dec/DECAY_BELLE2.DEC?at=refs%2Ftags%2Frelease-05-02-00#346,370,3874,3901
        The correlations are not published from this model... These look like someone took the correlations that Florian (?) calculated for the model below and just copied them twice...
        So this sounds fairly wrong...
        MOREOVER, THE EVTGEN BGL IMPLEMENTATION IS BASED ON A PAPER THAT IS WRONG!
        THE BLASCHKE??!?
        FACTORS ARE SET TO 1 THERE WHICH IS TOO EASY. HAMMER, HOWEVER, JUST HAS THE CORRECT BGL VERSION IN IT. SO IT IS NEEDED TO ADJUST HAMMERS BtoD BGL CLASS TO THE ONE THAT BELLE II USES AS AN INPUT:
        https://stash.desy.de/projects/B2/repos/basf2/browse/generators/evtgen/models/src/EvtBGLFF.cc#66,72,74
        """
        return cls(
            Xc="D",
            lep="Ell",
            params={
                "a+_1": uf(0.0126, 0.0001),
                "a+_2": uf(-0.094, 0.003),
                "a+_3": uf(0.34, 0.04),
                "a+_4": uf(-0.1, 0.6),
                "a0_1": uf(0.0115, 0.0001),
                "a0_2": uf(-0.057, 0.002),
                "a0_3": uf(0.12, 0.04),
                "a0_4": uf(0.4, 0.7),
            },
            param_init="{ap: [%f,%f,%f,%f], a0: [%f,%f,%f,%f]}",
            corr_matrix=np.array(
                [
                    [1.0, 0.245, -0.161, 0.020, 0.0, 0.0, 0.0, 0.0],
                    [0.245, 1.0, -0.654, 0.272, 0.0, 0.0, 0.0, 0.0],
                    [-0.161, -0.654, 1.0, -0.770, 0.0, 0.0, 0.0, 0.0],
                    [0.020, 0.272, -0.770, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.245, -0.161, 0.020],
                    [0.0, 0.0, 0.0, 0.0, 0.245, 1.0, -0.654, 0.272],
                    [0.0, 0.0, 0.0, 0.0, -0.161, -0.654, 1.0, -0.770],
                    [0.0, 0.0, 0.0, 0.0, 0.020, 0.272, -0.770, 1.0],
                ]
            ),
            expert_info=cls.BtoDEllNu_dec.__func__.__doc__,
        )

    @classmethod
    def BtoDEllNu_dec_updated(cls):
        """
        B -> D FFs from Tommy who got it from Christoph Schwanda. He redid the fit from arXiv:1510.03657 for another
        study, so this is the "official" covariance matrix. The central values changed slightly.
        there is also a covariance matrix provided by Christoph (Tommy's mail from Oct. 13, 2022)
        that can be used to compare and cross check.
        According to Tommy, a0_1 is extracted differently in the fit and thus probably
        uncorrelated to the rest. [Tommy's updated mail is from Dec. 16, 2022.]
        """

        return cls(
            Xc="D",
            lep="Ell",
            params={
                "a+_1": uf(0.0127, 0.0001),
                "a+_2": uf(-0.095, 0.003),
                "a+_3": uf(0.393, 0.157),
                "a+_4": uf(-0.577, 2.244),
                "a0_1": uf(0.0114, 0.0001),
                "a0_2": uf(-0.058, 0.003),
                "a0_3": uf(0.231, 0.139),
                "a0_4": uf(-0.789, 2.143),
            },
            param_init="{ap: [%f,%f,%f,%f], a0: [%f,%f,%f,%f]}",
            corr_matrix=np.array(
                [
                    [1.000, 0.130, -0.028, -0.042, 0.000, 0.122, 0.091, -0.108],
                    [0.130, 1.000, -0.575, 0.349, 0.000, 0.720, -0.170, 0.029],
                    [-0.028, -0.575, 1.000, -0.929, 0.000, -0.407, 0.649, -0.586],
                    [-0.042, 0.349, -0.929, 1.000, 0.000, 0.254, -0.690, 0.750],
                    [0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000],
                    [0.122, 0.720, -0.407, 0.254, 0.000, 1.000, -0.438, 0.244],
                    [0.091, -0.170, 0.649, -0.690, 0.000, -0.438, 1.000, -0.929],
                    [-0.108, 0.029, -0.586, 0.750, 0.000, 0.244, -0.929, 1.000],
                ]
            ),
            expert_info=cls.BtoDEllNu_dec_updated.__func__.__doc__,
        )

    @classmethod
    def BtoDEllNu_Scaled(cls, tau=False):
        """
        B -> D* FFs from Svenja (wg1_b2pilnu/Systematics/B2DstFF.ipynb), BGL(1,1,2) parameters of
        https://arxiv.org/pdf/2008.09341v1.pdf table VI, as used (already Vcb-scaled) in the dec file:
        https://stash.desy.de/projects/B2/repos/software/browse/decfiles/dec/DECAY_BELLE2.DEC?at=refs%2Ftags%2Frelease-05-02-00#345,369,3873,3900
        Only used for central values, so no correlations/errors needed; Vcb is 1.0/1.0066 but irrelevant here.
        EvtGen's BGL sets a0f=0 (equivalent to Hammer's d-vector=0), cf.
        https://stash.desy.de/projects/B2/repos/basf2/browse/generators/evtgen/models/src/EvtBGLFF.cc#141 vs
        https://gitlab.com/mpapucci/Hammer/-/blob/development/src/FormFactors/BGL/FFBtoDstarBGL.cc#L175
        Note: Hammer's own default parameters (https://arxiv.org/pdf/1902.09553.pdf) differ substantially from
        these Belle II inputs, for reasons not fully understood (possibly differing lattice QCD treatment or
        parameter definitions) - the shapes were cross-checked and agree in https://arxiv.org/pdf/1908.09398.pdf fig. 2.
        """

        return cls(
            Xc="D",
            lep="Ell" if not tau else "Tau",
            params={
                "ag_0": uf(0.02596, 0),
                "ag_1": uf(-0.06049, 0),
                "af_0": uf(0.01311, 0),
                "af_1": uf(0.01713, 0),
                "aF1_1": uf(0.00753, 0),
                "aF1_2": uf(-0.09346, 0),
            },
            param_init="{avec: [%f,%f], bvec: [%f,%f], cvec: [%f,%f], dvec: [0.0, 0.0], Vcb: 0.993443}",
            corr_matrix=np.identity(6),
            expert_info=cls.BtoDEllNu_Scaled.__func__.__doc__,
        )


class BLPRXP(FFModel):
    def __init__(
        self,
        Xc: str,
        lep: str,
        params: dict,
        param_init: dict,
        corr_matrix: np.ndarray | dict,
        expert_info: str,
    ):
        super().__init__(
            "BLPRXP", Xc, lep, params, param_init, corr_matrix, expert_info
        )

    @classmethod
    def BtoDstEllNu_outdated(cls):
        """
        compare with https://gitlab.com/mpapucci/Hammer/-/blob/master/src/FormFactors/FFBLPRBase.cc;
        https://arxiv.org/pdf/1703.05330.pdf, table II, L_w>=1+SR, corr: Table X
        Vcb cancels in nominator and denominator.
        paramters that occur in paper but not in Hammer: G(1), F(1).
        Wise versa: "a" = 1.509/sqrt(2), dV20 = 0, as = 0.26, la = 0.57115, ebR/eb = 0.861, ecR/ec = 0.822
        """

        return cls(
            Xc="D*",
            lep="Ell",
            params={
                "RhoSq": uf(1.24, 6.00e-02),
                "chi21": uf(-6.00e-02, 2.00e-02),
                "chi2p": uf(0.00, 2.00e-02),
                "chi3p": uf(5.00e-02, 2.00e-02),
                "eta1": uf(3.00e-01, 3.00e-02),
                "etap": uf(-5.00e-02, 9.00e-02),
                "mb": uf(4.71, 5.00e-02),
                "DelMbc": {"DelMbc": uf(3.40, 2.00e-02), "mc": uf(1.310, 0.00)},
            },
            param_init="{RhoSq: %f, chi21: %f, chi2p: %f, chi3p: %f, eta1: %f, etap: %f, mb: %f, mc: %f}",
            corr_matrix=np.array(
                [
                    [1.00, -0.13, -0.08, 0.72, 0.35, 0.13, -0.49, 0.05],
                    [-0.13, 1.00, -0.04, 0.25, -0.10, -0.11, 0.06, -0.01],
                    [-0.08, -0.04, 1.00, 0.07, -0.05, -0.06, 0.04, 0.00],
                    [0.72, 0.25, 0.07, 1.00, 0.16, 0.19, -0.06, 0.01],
                    [0.35, -0.10, -0.05, 0.16, 1.00, 0.05, -0.48, 0.05],
                    [0.13, -0.11, -0.06, 0.19, 0.05, 1.00, 0.04, 0.00],
                    [-0.49, 0.06, 0.04, -0.06, -0.48, 0.04, 1.00, 0.01],
                    [0.05, -0.01, 0.00, 0.01, 0.05, 0.00, 0.01, 1.00],
                ]
            ),
            expert_info=cls.BtoDstEllNu_outdated.__func__.__doc__,
        )

    # Shared fit result for BtoDstEllNu_new/BtoDEllNu_new: Florian said to take the first
    # column of https://arxiv.org/pdf/2206.11281.pdf table VII (L^{D;D*}_{w>=1;=1}), which
    # applies to both D and D* - same params/correlations (Table X), only Xc differs.
    _BLPRXP_NEW_DOC = """
        compare with https://gitlab.com/mpapucci/Hammer/-/blob/master/src/FormFactors/FFBLPRXPBase.cc;
        https://arxiv.org/pdf/2206.11281.pdf, table VII, first column (L^{D;D*}_{w>=1;=1}).
        Vcb cancels in nominator and denominator. Correlations: Table X
        THIS IS NOT PART OF HAMMERS OFFICIAL LAST VERSION YET! -> see development branch, there the central
        values agree
        """
    _BLPRXP_NEW_PARAMS = {
        "RhoStSq": uf(1.10, 0.04),
        "cSt": uf(2.39, 0.18),
        "mb": uf(4.71, 0.05),
        "DelMbc": {"DelMbc": uf(3.41, 0.02), "mc": uf(1.30, 0.00)},
        "la2": uf(0.12, 0.02),
        "eta1": uf(0.34, 0.04),
        "rho1": uf(-0.36, 0.24),
        "chi21": uf(-0.12, 0.02),
        "phi1p": uf(0.25, 0.21),
    }
    _BLPRXP_NEW_PARAM_INIT = "{RhoStSq: %f, cSt: %f, mb: %f, mc: %f, la2: %f, eta1: %f, rho1: %f, chi21: %f, phi1p: %f, chi2p: 0.0, chi3p: 0.0, etap: 0.0, beta21: 0.0, beta3p: 0.0}"
    _BLPRXP_NEW_CORR_MATRIX = np.array(
        [
            [1.000, 0.357, -0.720, 0.107, 0.034, 0.421, -0.075, -0.473, -0.632],
            [0.357, 1.000, -0.460, 0.048, -0.056, 0.383, -0.076, -0.647, -0.112],
            [-0.720, -0.460, 1.000, 0.028, 0.008, -0.429, -0.007, 0.369, 0.362],
            [0.107, 0.048, 0.028, 1.000, 0.009, 0.108, 0.477, -0.089, 0.011],
            [0.034, -0.056, 0.008, 0.009, 1.000, -0.255, -0.094, -0.034, -0.006],
            [0.421, 0.383, -0.429, 0.108, -0.255, 1.000, -0.379, -0.374, 0.189],
            [-0.075, -0.076, -0.007, 0.477, -0.094, -0.379, 1.000, 0.105, -0.279],
            [-0.473, -0.647, 0.369, -0.089, -0.034, -0.374, 0.105, 1.000, 0.305],
            [-0.632, -0.112, 0.362, 0.011, -0.006, 0.189, -0.279, 0.305, 1.000],
        ]
    )

    @classmethod
    def BtoDstEllNu_new(cls, tau: bool = False):
        return cls(
            Xc="D*",
            lep="Ell" if not tau else "Tau",
            params=cls._BLPRXP_NEW_PARAMS,
            param_init=cls._BLPRXP_NEW_PARAM_INIT,
            corr_matrix=cls._BLPRXP_NEW_CORR_MATRIX,
            expert_info=cls._BLPRXP_NEW_DOC,
        )

    @classmethod
    def BtoDEllNu_new(cls, tau: bool = False):
        return cls(
            Xc="D",
            lep="Ell" if not tau else "Tau",
            params=cls._BLPRXP_NEW_PARAMS,
            param_init=cls._BLPRXP_NEW_PARAM_INIT,
            corr_matrix=cls._BLPRXP_NEW_CORR_MATRIX,
            expert_info=cls._BLPRXP_NEW_DOC,
        )


class BLR(FFModel):
    def __init__(
        self,
        Xc: str,
        lep: str,
        params: dict,
        param_init: dict,
        corr_matrix: np.ndarray | dict,
        expert_info: str,
    ):
        super().__init__("BLR", Xc, lep, params, param_init, corr_matrix, expert_info)

    @classmethod
    def BtoD1EllNu_dec(cls, tau: bool = False):
        """
        B -> D_1 and B -> D_2* FFs from
        the dec file https://stash.desy.de/projects/B2/repos/software/browse/decfiles/dec/DECAY_BELLE2.DEC?at=refs%2Ftags%2Frelease-05-02-00#347,350
        as extrapolated from HAMMER: https://gitlab.com/mpapucci/Hammer/-/blob/master/src/FormFactors/FFBtoD1LLSW.cc#L54
        for mb and mc see https://stash.desy.de/projects/B2/repos/basf2/browse/generators/evtgen/models/src/EvtLLSWFF.cc#44
        Everybody always cites https://arxiv.org/pdf/hep-ph/9705467.pdf but these exact numbers are from
        https://arxiv.org/pdf/1606.09300.pdf table X approx C. Correlations are eq. 32 of the same paper.
        Note that the model BLR is still called "LLSW" in the EvtGen dec file while HAMMER already knows it as "BLR"!!
        EvtGen secretly also uses BLR already, however inconsistently, it does not use the alpha_s correction yet!
        HAMMER does not have BLR without alpha_s correction...
        these here are the narrow 3/2^+ states.
        """

        return cls(
            Xc="D**1",
            lep="Ell" if not tau else "Tau",
            params={
                "t1": uf(0.71, 0.07),
                "tp": uf(-1.60, 0.20),
                "tau1": uf(-0.50, 0.30),
                "tau2": uf(2.90, 1.60),
                # HAMMER also has laB=0.40, laP=0.80; not in the dec file, left at defaults
            },
            param_init="{t1: %f, tp: %f, tau1: %f, tau2: %f, mb: 4.2, mc: 1.4}",
            corr_matrix=np.array(
                [
                    [1.00, -0.83, 0.66, -0.63],
                    [-0.83, 1.00, -0.27, 0.20],
                    [0.66, -0.27, 1.00, -0.93],
                    [-0.63, 0.20, -0.93, 1.00],
                ]
            ),
            expert_info=cls.BtoD1EllNu_dec.__func__.__doc__,
        )

    @classmethod
    def BtoD2stEllNu_dec(cls, tau: bool = False):
        """
        B -> D_1 and B -> D_2* FFs from
        the dec file https://stash.desy.de/projects/B2/repos/software/browse/decfiles/dec/DECAY_BELLE2.DEC?at=refs%2Ftags%2Frelease-05-02-00#347,350
        as extrapolated from HAMMER: https://gitlab.com/mpapucci/Hammer/-/blob/master/src/FormFactors/FFBtoD1LLSW.cc#L54
        for mb and mc see https://stash.desy.de/projects/B2/repos/basf2/browse/generators/evtgen/models/src/EvtLLSWFF.cc#44
        Everybody always cites https://arxiv.org/pdf/hep-ph/9705467.pdf but these exact numbers are from
        https://arxiv.org/pdf/1606.09300.pdf table X approx C. Correlations are eq. 32 of the same paper.
        Note that the model BLR is still called "LLSW" in the EvtGen dec file while HAMMER already knows it as "BLR"!!
        EvtGen secretly also uses BLR already, however inconsistently, it does not use the alpha_s correction yet!
        HAMMER does not have BLR without alpha_s correction...
        these here are the narrow 3/2^+ states.
        """

        return cls(
            Xc="D**2*",
            lep="Ell" if not tau else "Tau",
            params={
                "t1": uf(0.71, 0.07),
                "tp": uf(-1.60, 0.20),
                "tau1": uf(-0.50, 0.30),
                "tau2": uf(2.90, 1.60),
                # HAMMER also has laB=0.40, laP=0.80; not in the dec file, left at defaults
            },
            param_init="{t1: %f, tp: %f, tau1: %f, tau2: %f, mb: 4.2, mc: 1.4}",
            corr_matrix=np.array(
                [
                    [1.00, -0.83, 0.66, -0.63],
                    [-0.83, 1.00, -0.27, 0.20],
                    [0.66, -0.27, 1.00, -0.93],
                    [-0.63, 0.20, -0.93, 1.00],
                ]
            ),
            expert_info=cls.BtoD2stEllNu_dec.__func__.__doc__,
        )

    @classmethod
    def BtoD0stEllNu_dec(cls, tau: bool = False):
        """
        B -> D_0* and B -> D'_1 FFs from the dec file https://stash.desy.de/projects/B2/repos/software/browse/decfiles/dec/DECAY_BELLE2.DEC?at=refs%2Ftags%2Frelease-05-02-00#348-349
        https://gitlab.com/mpapucci/Hammer/-/blob/master/src/FormFactors/FFBtoD0starLLSW.cc#L54
        for mb and mc see https://stash.desy.de/projects/B2/repos/basf2/browse/generators/evtgen/models/src/EvtLLSWFF.cc#27
        Everybody always cites https://arxiv.org/pdf/hep-ph/9705467.pdf but these exact numbers are from
        https://arxiv.org/pdf/1606.09300.pdf table X approx C. Correlations are eq. 33 of the same paper.
        Note that the model BLR is still called "LLSW" in the EvtGen dec file while HAMMER already knows it as "BLR"!!
        EvtGen secretly also uses BLR already, however inconsistently, it does not use the alpha_s correction yet!
        HAMMER does not have BLR without alpha_s correction...
        these here are the broad 1/2^+ states.
        """

        return cls(
            Xc="D**0*",
            lep="Ell" if not tau else "Tau",
            params={
                "zt1": uf(0.68, 0.20),
                "ztp": uf(-0.20, 1.20),
                "zeta1": uf(0.30, 0.30),
                # HAMMER also has laB=0.40, laS=0.76; not in the dec file, left at defaults
            },
            param_init="{zt1: %f, ztp: %f, zeta1: %f, mb: 4.2, mc: 1.4}",
            corr_matrix=np.array(
                [
                    [1.00, -0.95, -0.35],
                    [-0.95, 1.00, 0.51],
                    [-0.35, 0.51, 1.00],
                ]
            ),
            expert_info=cls.BtoD0stEllNu_dec.__func__.__doc__,
        )

    @classmethod
    def BtoD1prEllNu_dec(cls, tau: bool = False):
        """
        B -> D_0* and B -> D'_1 FFs from the dec file https://stash.desy.de/projects/B2/repos/software/browse/decfiles/dec/DECAY_BELLE2.DEC?at=refs%2Ftags%2Frelease-05-02-00#348-349
        https://gitlab.com/mpapucci/Hammer/-/blob/master/src/FormFactors/FFBtoD0starLLSW.cc#L54
        for mb and mc see https://stash.desy.de/projects/B2/repos/basf2/browse/generators/evtgen/models/src/EvtLLSWFF.cc#27
        Everybody always cites https://arxiv.org/pdf/hep-ph/9705467.pdf but these exact numbers are from
        https://arxiv.org/pdf/1606.09300.pdf table X approx C. Correlations are eq. 33 of the same paper.
        Note that the model BLR is still called "LLSW" in the EvtGen dec file while HAMMER already knows it as "BLR"!!
        EvtGen secretly also uses BLR already, however inconsistently, it does not use the alpha_s correction yet!
        HAMMER does not have BLR without alpha_s correction...
        these here are the broad 1/2^+ states.
        """

        return cls(
            Xc="D**1*",
            lep="Ell" if not tau else "Tau",
            params={
                "zt1": uf(0.68, 0.20),
                "ztp": uf(-0.20, 1.20),
                "zeta1": uf(0.30, 0.30),
                # HAMMER also has laB=0.40, laS=0.76; not in the dec file, left at defaults
            },
            param_init="{zt1: %f, ztp: %f, zeta1: %f, mb: 4.2, mc: 1.4}",
            corr_matrix=np.array(
                [
                    [1.00, -0.95, -0.35],
                    [-0.95, 1.00, 0.51],
                    [-0.35, 0.51, 1.00],
                ]
            ),
            expert_info=cls.BtoD1prEllNu_dec.__func__.__doc__,
        )

    @classmethod
    def BtoD1EllNu_new(cls, tau: bool = False):
        """
        New values from https://arxiv.org/pdf/1711.03110.pdf table V, including the alpha-s
        variation; an update of https://arxiv.org/pdf/1606.09300.pdf. mb, mc taken from HAMMER:
        https://gitlab.com/mpapucci/Hammer/-/blob/master/src/FormFactors/FFBtoD1BLR.cc#L54-55
        """

        return cls(
            Xc="D**1",
            lep="Ell" if not tau else "Tau",
            params={
                "t1": uf(0.70, 0.07),
                "tp": uf(-1.60, 0.20),
                "tau1": uf(-0.50, 0.30),
                "tau2": uf(2.90, 1.40),
                # HAMMER also has laB=0.40, laP=0.80; not in the dec file, left at defaults
            },
            param_init="{t1: %f, tp: %f, tau1: %f, tau2: %f, mb: 4.71, mc: 1.31}",
            corr_matrix=np.array(
                [
                    [1.00, -0.85, 0.53, -0.49],
                    [-0.85, 1.00, -0.17, 0.086],
                    [0.53, -0.17, 1.00, -0.89],
                    [-0.49, 0.086, -0.89, 1.00],
                ]
            ),
            expert_info=cls.BtoD1EllNu_new.__func__.__doc__,
        )

    @classmethod
    def BtoD2stEllNu_new(cls, tau: bool = False):
        """
        New values from https://arxiv.org/pdf/1711.03110.pdf table V, including the alpha-s
        variation; an update of https://arxiv.org/pdf/1606.09300.pdf. mb, mc taken from HAMMER:
        https://gitlab.com/mpapucci/Hammer/-/blob/master/src/FormFactors/FFBtoD1BLR.cc#L54-55
        """

        return cls(
            Xc="D**2*",
            lep="Ell" if not tau else "Tau",
            params={
                "t1": uf(0.70, 0.07),
                "tp": uf(-1.60, 0.20),
                "tau1": uf(-0.50, 0.30),
                "tau2": uf(2.90, 1.40),
                # HAMMER also has laB=0.40, laP=0.80; not in the dec file, left at defaults
            },
            param_init="{t1: %f, tp: %f, tau1: %f, tau2: %f, mb: 4.71, mc: 1.31}",
            corr_matrix=np.array(
                [
                    [1.00, -0.85, 0.53, -0.49],
                    [-0.85, 1.00, -0.17, 0.086],
                    [0.53, -0.17, 1.00, -0.89],
                    [-0.49, 0.086, -0.89, 1.00],
                ]
            ),
            expert_info=cls.BtoD2stEllNu_new.__func__.__doc__,
        )

    @classmethod
    def BtoD0stEllNu_new(cls, tau: bool = False):
        """
        New values from https://arxiv.org/pdf/1711.03110.pdf table V, including the alpha-s
        variation; an update of https://arxiv.org/pdf/1606.09300.pdf. mb, mc taken from HAMMER:
        https://gitlab.com/mpapucci/Hammer/-/blob/master/src/FormFactors/FFBtoD1BLR.cc#L54-55
        """

        return cls(
            Xc="D**0*",
            lep="Ell" if not tau else "Tau",
            params={
                "zt1": uf(0.70, 0.21),
                "ztp": uf(0.20, 1.40),
                "zeta1": uf(0.60, 0.30),
                # HAMMER also has laB=0.40, laS=0.76; not in the dec file, left at defaults
            },
            param_init="{zt1: %f, ztp: %f, zeta1: %f, mb: 4.71, mc: 1.31}",
            corr_matrix=np.array(
                [
                    [1.00, -0.95, -0.44],
                    [-0.95, 1.00, 0.61],
                    [-0.44, 0.61, 1.00],
                ]
            ),
            expert_info=cls.BtoD0stEllNu_new.__func__.__doc__,
        )

    @classmethod
    def BtoD1prEllNu_new(cls, tau: bool = False):
        """
        New values from https://arxiv.org/pdf/1711.03110.pdf table V, including the alpha-s
        variation; an update of https://arxiv.org/pdf/1606.09300.pdf. mb, mc taken from HAMMER:
        https://gitlab.com/mpapucci/Hammer/-/blob/master/src/FormFactors/FFBtoD1BLR.cc#L54-55
        """

        return cls(
            Xc="D**1*",
            lep="Ell" if not tau else "Tau",
            params={
                "zt1": uf(0.70, 0.21),
                "ztp": uf(0.20, 1.40),
                "zeta1": uf(0.60, 0.30),
                # HAMMER also has laB=0.40, laS=0.76; not in the dec file, left at defaults
            },
            param_init="{zt1: %f, ztp: %f, zeta1: %f, mb: 4.71, mc: 1.31}",
            corr_matrix=np.array(
                [
                    [1.00, -0.95, -0.44],
                    [-0.95, 1.00, 0.61],
                    [-0.44, 0.61, 1.00],
                ]
            ),
            expert_info=cls.BtoD1prEllNu_new.__func__.__doc__,
        )


class ISGW2(FFModel):
    def __init__(
        self,
        Xc: str,
        lep: str,
        params: dict,
        param_init: dict,
        corr_matrix: np.ndarray | dict,
        expert_info: str,
    ):
        super().__init__("ISGW2", Xc, lep, params, param_init, corr_matrix, expert_info)

    @classmethod
    def BtoD1EllNu_dec(cls):
        """
        ISGW2 works without input parameters in Hammer and in EvtGen. Only serves as an input.
        """

        return cls(
            Xc="D**0",
            lep="Ell",
            params={},
            param_init=None,
            corr_matrix=None,
            expert_info=cls.BtoD1EllNu_dec.__func__.__doc__,
        )


class CLN(FFModel):
    def __init__(
        self,
        Xc: str,
        lep: str,
        params: dict,
        param_init: dict,
        corr_matrix: np.ndarray | dict,
        expert_info: str,
    ):
        super().__init__("CLN", Xc, lep, params, param_init, corr_matrix, expert_info)

    @classmethod
    def BtoDTauNu_dec(cls):
        """
        https://arxiv.org/pdf/1612.07233.pdf, eq. 189, 191, corr: 190
        a is constantly treated as 1. in EvtGen as seen when comparing https://gitlab.com/mpapucci/Hammer/-/blob/master/src/FormFactors/FFBtoDCLN.cc#L84
        to https://stash.desy.de/projects/B2/repos/basf2/browse/generators/evtgen/models/src/EvtHQET3FF.cc#54
        G1 is indeed what EvtGen calls "v1_1" as seen when comparing lines 100-102 (Hammer) with 55-57 (EvtGen)
        """

        return cls(
            Xc="D",
            lep="Tau",
            params={
                "G1": uf(1.0541, 0.0083),
                "RhoSq": uf(1.1280, 0.0330),
                "Delta": uf(1.0000, 0),
            },
            param_init="{G1: %f, RhoSq: %f, Delta: %f}",
            corr_matrix=np.array(
                [
                    [1.000, 0.751, 0.000],
                    # Vcb assumed a constant factor that does not affect the correlation
                    [0.751, 1.000, 0.000],
                    [0.000, 0.000, 1.000],  # TODO: Delta's correlation to G1/RhoSq is unknown
                ]
            ),
            expert_info=cls.BtoDTauNu_dec.__func__.__doc__,
        )

    @classmethod
    def BtoDstTauNu_dec(cls):
        """
        https://arxiv.org/pdf/1612.07233.pdf, p.138
        a is constantly treated as 1. in EvtGen as seen when comparing https://gitlab.com/mpapucci/Hammer/-/blob/master/src/FormFactors/FFBtoDCLN.cc#L84
        to https://stash.desy.de/projects/B2/repos/basf2/browse/generators/evtgen/models/src/EvtHQET3FF.cc#54
        G1 is indeed what EvtGen calls "v1_1" as seen when comparing lines 100-102 (Hammer) with 55-57 (EvtGen)
        """

        return cls(
            Xc="D*",
            lep="Tau",
            params={
                "F1": uf(0.912, 0.014),
                "RhoSq": uf(1.205, 0.026),
                "R0": uf(1.150, 0),  # TODO: source/uncertainty for R0 not found yet
                "R1": uf(1.404, 0.032),
                "R2": uf(0.854, 0.020),
            },
            param_init="{F1: %f, RhoSq: %f, R0: %f, R1: %f, R2: %f}",
            corr_matrix=np.array(
                [
                    [1.000, 0.338, 0.000, -0.104, -0.071],
                    # Vcb assumed a constant factor that does not affect the correlation
                    [0.338, 1.000, 0.000, 0.570, -0.810],
                    [0.000, 0.000, 1.000, 0.000, 0.000],  # TODO: R0's correlation to others is unknown
                    [-0.104, 0.570, 0.000, 1.000, -0.758],
                    [-0.071, -0.810, 0.000, -0.758, 1.000],
                ]
            ),
            expert_info=cls.BtoDstTauNu_dec.__func__.__doc__,
        )


class FFModelVisualizer(Visualizer):
    def __init__(self, instance: FFModel):
        super().__init__(instance)

    def annotate_matrix_plot(self, ax: plt.Axes):

        if isinstance(ax, plt.Axes):
            self._annotate_axis(ax)
        elif isinstance(ax, np.ndarray):
            for axis in ax:
                self._annotate_axis(axis)

    def _annotate_axis(self, ax):

        ax.set_xlabel("FF model parameters")
        ax.set_ylabel("FF model parameters")

        ax.set_xticks(
            np.arange(self.instance.num_params) + 0.5,
            self.instance.params.keys(),
        )
        ax.set_yticks(
            np.arange(self.instance.num_params) + 0.5,
            self.instance.params.keys(),
        )

    def plot_cov_matrix(
        self,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        save: bool = False,
        filename: str = "",
    ) -> tuple[plt.Figure, plt.Axes]:

        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
        elif fig is None or ax is None:
            raise ValueError("You must provide both fig and ax or none of them.")

        sns.heatmap(
            self.instance.cov_matrix,
            ax=ax,
            annot=True,
            cbar_kws={"label": "Covariance"},
            cmap=CMAP,
        )
        ax.set_title("Covariance matrix")
        self.annotate_matrix_plot(ax)

        if save:
            self.instance.saving_info["namespace"] = ["ff_model", "cov_matrix"]
            self.save_figure(fig, filename)

        return fig, ax

    def plot_corr_matrix(
        self,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        save: bool = False,
        filename: str = "",
    ) -> tuple[plt.Figure, plt.Axes]:

        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
        elif fig is None or ax is None:
            raise ValueError("You must provide both fig and ax or none of them.")

        sns.heatmap(
            self.instance.corr_matrix,
            ax=ax,
            annot=True,
            cbar_kws={"label": "Pearson coeff."},
            cmap=CMAP,
            vmin=0,
            vmax=1,
        )

        ax.set_title(
            " ".join(
                (
                    "Correlation matrix",
                    self.instance.name,
                    r"$\mathrm{B} \rightarrow$",
                    get_latex_symbol(self.instance.Xc),
                    get_latex_symbol(self.instance.lep),
                    get_latex_symbol("Nu"),
                )
            )
        )
        if save:
            self.instance.saving_info["namespace"] = ["ff_model", "corr_matrix"]
            self.save_figure(fig, filename)

        return fig, ax

    def plot_params(
        self,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        save: bool = False,
        filename: str = "",
    ) -> tuple[plt.Figure, plt.Axes]:

        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
        elif fig is None or ax is None:
            raise ValueError("You must provide both fig and ax or none of them.")

        ax.errorbar(
            x=np.arange(self.instance.num_params) + 0.5,
            y=self.instance.parameter_central_values,
            yerr=self.instance.parameter_errors,
            linestyle="",
            color="black",
            marker=".",
        )

        ax.table(
            cellText=[
                [round(x, 4) for x in self.instance.parameter_central_values],
                [round(x, 4) for x in self.instance.parameter_errors],
            ],
            rowLabels=["value", "unc"],
            colLabels=[
                x if x != "DelMbc" else "DelMbc(mc)"
                for x in self.instance.params.keys()
            ],
            loc="top",
        )

        ax.set_xticks(
            np.arange(self.instance.num_params) + 0.5,
            self.instance.params.keys(),
        )
        ax.set_xlabel("FF model parameters")
        ax.set_ylabel("Value")

        if save:
            self.instance.saving_info["namespace"] = ["ff_model", "params"]
            self.save_figure(fig, filename)

        return fig, ax

    def plot_corr_and_params(self, save: bool = False, filename: str = ""):

        fig, ax = plt.subplots(1, 2, figsize=(16, 4.5), dpi=DPI)

        self.plot_params(fig=fig, ax=ax[0])

        self.plot_corr_matrix(fig=fig, ax=ax[1])
        self.annotate_matrix_plot(ax[1])
        if save:
            self.instance.saving_info["namespace"] = [
                "ff_model",
                "corr_matrix_and_params",
            ]
            self.save_figure(fig, filename)

        return fig, ax
