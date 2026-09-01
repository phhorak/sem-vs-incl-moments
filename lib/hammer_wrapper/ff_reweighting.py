from __future__ import annotations

import logging
import numpy as np

from pandas import DataFrame
from collections.abc import MutableMapping
from typing import Iterable, Dict, List, Iterator, Tuple

from sysvar.utils import corr2cov

from hammer.hammerlib import Hammer, Process, Particle, FourMomentum

try:
    from rdstar1prong.hammer_wrapper.ff_models import (
        FFModel,
        BGL,
        BGLB2,
        CLN,
        BLPRXP,
        BLR,
        print_model_info,
    )
except ImportError:
    from hammer_wrapper.ff_models import (
        FFModel,
        BGL,
        BGLB2,
        CLN,
        BLPRXP,
        BLR,
        print_model_info,
    )


logging.basicConfig(
    format="%(levelname)s : %(funcName)s: %(lineno)d :  %(message)s",
    level=logging.INFO,
)


class HammerRateContainer(MutableMapping):
    def __init__(self) -> None:
        self._rates_dict: Dict[str, float] = dict()

    def __getitem__(self, item: str) -> float:
        return self._rates_dict[item]

    def __setitem__(self, key: str, value: float):
        self._rates_dict[key] = value

    def __delitem__(self, key) -> None:
        del self._rates_dict[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._rates_dict.keys())

    def __len__(self) -> int:
        return len(self._rates_dict)

    def get_or_calculate_rate(
        self,
        scheme_name: str,
        hammer_id_str: str,
        hammer_instance: Hammer,
    ) -> float:
        key: str = f"{scheme_name}_{hammer_id_str}"
        try:
            return self._rates_dict[key]
        except KeyError:
            rate: float = hammer_instance.get_rate(hammer_id_str, scheme_name)
            assert isinstance(rate, float), (
                rate,
                type(rate),
                hammer_id_str,
                scheme_name,
                key,
            )
            self._rates_dict[key] = rate
            return self._rates_dict[key]

    def get_or_calculate_input_rate(
        self,
        hammer_id_str: str,
        hammer_instance: Hammer,
    ) -> float:
        key: str = f"INPUT_{hammer_id_str}"
        try:
            return self._rates_dict[key]
        except KeyError:
            input_rate: float = hammer_instance.get_denominator_rate(hammer_id_str)
            assert isinstance(input_rate, float), (
                input_rate,
                type(input_rate),
                hammer_id_str,
                key,
            )
            self._rates_dict[key] = input_rate
            return self._rates_dict[key]


class HammerStudy:
    def __init__(self, mode: str, default_names: bool = True):

        if mode == "B2":
            self.input_models = [
                BGL.BtoDstEllNu_B2dec(),
                BGLB2.BtoDEllNu_dec(),
                BLR.BtoD1EllNu_dec(),
                BLR.BtoD0stEllNu_dec(),
                BLR.BtoD1prEllNu_dec(),
                BLR.BtoD2stEllNu_dec(),
                CLN.BtoDTauNu_dec(),
                CLN.BtoDstTauNu_dec(),
                BLR.BtoD1EllNu_dec(tau=True),
                BLR.BtoD0stEllNu_dec(tau=True),
                BLR.BtoD1prEllNu_dec(tau=True),
                BLR.BtoD2stEllNu_dec(tau=True),
            ]
        else:
            raise NotImplementedError(
                "Not having default options for Belle and LHCb MC yet"
            )

        if default_names:
            self._generations = {
                "B": "B_anc",
                "B_daughters": [
                    "B_d0",
                    "B_d1",
                    "B_d2",
                ],
                "Xc_daughters": ["B_g_d0", "B_g_d1"],
                "tau_daughters": [
                    "B_g_d10",
                    "B_g_d11",
                    "B_g_d12",
                    "B_g_d13",
                    "B_g_d14",
                ],
                "Xc_grand_daughters": ["B_g_g_d0", "B_g_g_d1"],
                "Xc_great_grand_daughters": ["B_g_g_g_d0", "B_g_g_g_d1"],
            }
            self._four_momentum_vars = ["mcE", "mcPX", "mcPY", "mcPZ"]
            self._pdg_var = "mcPDG"

        else:
            self._generations = {
                "B": "",
                "B_daughters": [],
                "Xc_daughters": [],
                "tau_daughters": [],
                "Xc_grand_daughters": [],
                "Xc_great_grand_daughters": [],
            }
            self._four_momentum_vars = []
            self._pdg_var = []

        self.hammer = Hammer()
        self.rate_container = HammerRateContainer()

        self.histograms = {}

        self._print_information_setup()

    def _print_information_setup(self):

        logging.info("Set up Hammer study")
        logging.info("The input variables have been specified as follows:")
        logging.info("PDG variable: %s", self._pdg_var)
        logging.info(
            "Four momentum variables: %s, %s, %s, %s,",
            *self._four_momentum_vars,
        )
        logging.info(
            "B meson prefix: %s ",
            self._generations["B"],
        )
        logging.info(
            "B meson daughter prefixes: %s ",
            self._generations["B_daughters"],
        )
        logging.info(
            "Xc meson daughter prefixes: %s ",
            self._generations["Xc_daughters"],
        )
        logging.info(
            "The HammerStudy class will take care of building the full variable names"
        )
        logging.warning(
            "If this does not align with the variable names in your tuples set default_names = False in the object constructor and modify the following properties: %s",
            [
                key
                for key, attr in HammerStudy.__dict__.items()
                if (isinstance(attr, property) and attr.fset is not None)
            ],
        )
        logging.warning(
            "DO NOT CHANGE the names of the keys in the generations dictionary as this will cause some of the HammerStudy functionalities to fail"
        )

    @property
    def four_momentum_vars(self):
        return self._four_momentum_vars

    @four_momentum_vars.setter
    def four_momentum_vars(self, variables: Iterable):
        self._four_momentum_vars = variables

    @property
    def pdg_var(self):
        return self._pdg_var

    @pdg_var.setter
    def pdg_var(self, pdg: str):
        self._pdg_var = pdg

    @property
    def decays(self):
        return [self._get_decay_name(model) for model in self.input_models]

    @property
    def generations(self):
        return self._generations

    @generations.setter
    def generations(self, dictionary: dict):
        self._generations = dictionary

    @staticmethod
    def _get_ff_scheme_dictionary(model: FFModel, suffix: None | str = None):

        if suffix is None:
            dictionary = {f"B{model.Xc}": model.name}
        else:
            dictionary = {f"B{model.Xc}": f"{model.name}_{suffix}"}

        return dictionary

    @staticmethod
    def _get_decay_name(model: FFModel, to: bool = False):

        return f"Bto{model.Xc}{model.lep}Nu" if to else f"B{model.Xc}{model.lep}Nu"

    def set_ff_input_scheme(self, verbose=False):
        """
        This has the same name as the hammer API to ensure compatibility
        """

        logging.info("Defaulting to Belle II decay.dec parameters")
        if not verbose:
            logging.info(
                "Enable the verbose option when calling set_ff_input_scheme to know more about the models that were use as an input"
            )

        input_schemes_for_hammer = {}
        for model in self.input_models:
            model_scheme = self._get_ff_scheme_dictionary(model)
            input_schemes_for_hammer.update(model_scheme)
            self._add_Dstst_ff_models(model, input_schemes_for_hammer)

        self.hammer.set_ff_input_scheme(input_schemes_for_hammer)

        for model in self.input_models:
            param_string = self._create_param_string(model)
            if verbose:
                logging.info(
                    "Adding transition and model: %s as input for HAMMER", model_scheme
                )
                print_model_info(model)
                logging.info(
                    "The parameters of the model have been set as: %s: ", param_string
                )

            self.hammer.set_options(param_string)

    def _create_param_string(
        self,
        model: FFModel,
        suffix: None | str = None,
        parameters: None | Iterable = None,
    ):
        # HAMMER expects "BtoD*..." not "BD*..." for the input scheme, even though
        # the scheme dictionaries elsewhere use the "BD*" form without "to".
        if suffix is None:
            template = f"Bto{model.Xc}{model.name}: {model.param_init}"
        else:
            template = f"Bto{model.Xc}{model.name}_{suffix}: {model.param_init}"

        if parameters is None:
            string = template % tuple(
                (
                    param.nominal_value
                    if name != "DelMbc"
                    else param["mc"].nominal_value
                )
                for name, param in model.params.items()
            )
        else:
            string = f"Bto{model.Xc}{model.name}_{suffix}: {model.param_init}" % tuple(
                parameters
            )
        return string

    def add_ff_scheme(self, models, scheme_name, verbose=False):
        """
        This has the same name as the hammer API to ensure compatibility
        """

        scheme_dict = {scheme_name: {}}
        scheme_options = {scheme_name: []}

        for i, model in enumerate(models):
            if not isinstance(model, FFModel):
                raise ValueError(
                    f"Model number {i} in the arguments is not an instance of FFModel"
                )
            decay = self._get_decay_name(model)
            self.hammer.include_decay(decay)

            param_string = self._create_param_string(model=model, suffix=scheme_name)
            if verbose:
                logging.info("Adding %s model for %s decay", model.name, decay)
                print_model_info(model)
                logging.info("The parameters are set to: %s", param_string)

            # Use a scheme-specific model token so the central scheme parameters
            # (set below with suffix=scheme_name) are actually picked up by Hammer.
            scheme_dict[scheme_name].update(
                self._get_ff_scheme_dictionary(model, scheme_name)
            )
            scheme_options[scheme_name].append(param_string)

            self._add_Dstst_ff_models(model, scheme_dict[scheme_name])

            eigenvalue_variations = get_varied_FF_central_values(model)
            if verbose:
                logging.info(
                    "Now creating eigenvariations of the parameters %s",
                    eigenvalue_variations,
                )
            for i, eigenvar in enumerate(eigenvalue_variations):
                up_var, down_var = f"up{i}", f"down{i}"
                self._add_eigenvariation_to_scheme_setup(
                    scheme_dict, scheme_options, scheme_name, model, up_var, eigenvar[0]
                )
                self._add_eigenvariation_to_scheme_setup(
                    scheme_dict,
                    scheme_options,
                    scheme_name,
                    model,
                    down_var,
                    eigenvar[1],
                )

        if verbose:
            logging.info(
                "The scheme and it's eigenvariations have been set to: %s", scheme_dict
            )
            logging.info(
                "Now will pad the models with missing eigenvariations to ensure that hammer runs"
            )

        self._pad_missing_eigenvariations(
            models, scheme_dict, scheme_options, scheme_name
        )

        for (scheme_n1, names), options in zip(
            scheme_dict.items(), scheme_options.values()
        ):
            # make sure that all models have the D** decay form factors
            # This is added for the eigenvariations that are padded just before
            for model in models:
                self._add_Dstst_ff_models(model, names)

            self.hammer.add_ff_scheme(scheme_n1, names)
            for option in options:
                self.hammer.set_options(option)

    def _pad_missing_eigenvariations(
        self, models, scheme_dict: dict, scheme_options: dict, scheme_name: str
    ):

        for model in models:
            for (scheme_n1, model_names), options in zip(
                scheme_dict.items(), scheme_options.values()
            ):
                # Models can have differing numbers of eigenvariations; pad the
                # missing ones with the nominal value so every scheme has an
                # entry for every model, or HAMMER errors out.
                if f"B{model.Xc}" not in model_names.keys():
                    logging.warning(
                        "Will pad model %s for variation %s with the nominal values",
                        model.name,
                        scheme_n1,
                    )

                    model_names.update(
                        self._get_ff_scheme_dictionary(model, scheme_name)
                    )
                    options.append(
                        self._create_param_string(model=model, suffix=scheme_name)
                    )

    def _add_eigenvariation_to_scheme_setup(
        self,
        scheme_dict: dict,
        scheme_options: dict,
        scheme_name: str,
        model,
        var_token: str,
        parameters: np.ndarray,
    ):

        if scheme_name + "_" + var_token not in scheme_dict.keys():
            self._add_placeholder_for_variation(
                scheme_dict, scheme_options, scheme_name, var_token
            )

        scheme_dict[scheme_name + "_" + var_token].update(
            self._get_ff_scheme_dictionary(model, var_token)
        )
        scheme_options[scheme_name + "_" + var_token].append(
            self._create_param_string(
                model=model,
                suffix=var_token,
                parameters=parameters,
            )
        )

    @staticmethod
    def _add_placeholder_for_variation(
        scheme_dict: dict,
        scheme_options: dict,
        scheme_name: str,
        var_token: str,
    ):
        """
        This will create a subdictionary for a new variation
        """

        scheme_dict.update({scheme_name + "_" + var_token: {}})
        scheme_options.update({scheme_name + "_" + var_token: []})

    @staticmethod
    def _add_Dstst_ff_models(model, dictionary):

        if "D**" in model.Xc:
            dictionary.update({f"{model.Xc}D*Pi": "PW"})
            dictionary.update({f"{model.Xc}DPi": "PW"})

    def process_events(
        self,
        df: DataFrame,
        scheme: str,
        histogram_name: str = None,
        verbose=False,
    ):

        self.hammer.set_units("GeV")  # this is also the default
        self.hammer.init_run()

        df[self._get_new_df_columnnames(scheme)] = df.apply(
            lambda row: self._event_wise(row, scheme, histogram_name, verbose),
            axis=1,
            result_type="expand",
        )

    def _get_new_df_columnnames(self, scheme: str):

        new_cols = list(self.hammer.get_ff_scheme_names())
        new_cols.append(f"denom_rate_{scheme}")
        new_cols.append(f"new_rate_{scheme}")
        new_cols.append(f"hammer_reweighted_{scheme}")
        new_cols.append(f"hammer_found_rates_{scheme}")

        return new_cols

    def _event_wise(
        self,
        row,
        schemename: str,
        histogram_name: str = None,
        verbose: bool = False,
    ):

        # Not a semileptonic event: nothing to reweight.
        if row[
            f"{self.generations['B_daughters'][1]}_mcPDG"
        ] not in self._get_particle_pdg_codes("leptons"):
            return self._add_unweighted_results()

        # Tau daughters not properly matched in truth: skip reweighting rather than crash.
        if row[f"{self.generations['B_daughters'][1]}_mcPDG"] in [15, -15]:
            if (
                row[f"{self.generations['tau_daughters'][0]}_mcPDG"] == -9999
                and row[f"{self.generations['tau_daughters'][1]}_mcPDG"] == -9999
                and row[f"{self.generations['tau_daughters'][2]}_mcPDG"] == -9999
                and row[f"{self.generations['tau_daughters'][3]}_mcPDG"] == -9999
                and row[f"{self.generations['tau_daughters'][4]}_mcPDG"] == -9999
            ):
                return self._add_unweighted_results()

        # Build the decay tree generation by generation (breadth-first): a HAMMER
        # vertex can only reference particles already added to the process.
        self.hammer.init_event()
        process = Process()

        if not self._has_valid_particle(row, self._generations["B"]):
            return self._add_unweighted_results()
        Bmeson = self._add_hammer_particle(process, row, self._generations["B"])

        if not all(
            self._has_valid_particle(row, prefix)
            for prefix in self.generations["B_daughters"]
        ):
            return self._add_unweighted_results()
        B_daughters = [
            self._add_hammer_particle(process, row, prefix)
            for prefix in self.generations["B_daughters"]
        ]
        process.add_vertex(Bmeson, B_daughters)

        # Only D*/D** decays have an Xc sub-vertex.
        if row[f"{self._generations['B_daughters'][0]}_{self._pdg_var}"] in [
            *self._get_particle_pdg_codes("D*"),
            *self._get_particle_pdg_codes("D**"),
        ]:
            # Some tuples do not store Xc sub-decay truth daughters for all events.
            # If missing, still reweight the parent B->Xc l nu process.
            if all(
                self._has_valid_particle(row, prefix)
                for prefix in self.generations["Xc_daughters"]
            ):
                Xc_daughters = [
                    self._add_hammer_particle(process, row, prefix)
                    for prefix in self.generations["Xc_daughters"]
                ]
                process.add_vertex(B_daughters[0], Xc_daughters)
            else:
                Xc_daughters = []

        if row[f"{self._generations['B_daughters'][1]}_{self._pdg_var}"] in [
            *self._get_particle_pdg_codes("tau"),
        ]:
            tau_daughters = [
                self._add_hammer_particle(process, row, prefix)
                for prefix in self.generations["tau_daughters"]
                if row[f"{prefix}_mcPDG"] != -9999
            ]
            # Exclude tau -> pi pi pi nu: HAMMER can't contract the rate tensor for
            # this vertex (errors out with "uncontracted indices" / "Eject! Eject!").
            if (
                abs(row[f"{self._generations['tau_daughters'][0]}_{self._pdg_var}"])
                == 211
                and abs(row[f"{self._generations['tau_daughters'][1]}_{self._pdg_var}"])
                == 211
                and abs(row[f"{self._generations['tau_daughters'][2]}_{self._pdg_var}"])
                == 211
                and abs(row[f"{self._generations['tau_daughters'][3]}_{self._pdg_var}"])
                == 16
            ):
                pass
            else:
                process.add_vertex(B_daughters[1], tau_daughters)

        # D** decays have a further Xc grand-daughter vertex.
        if row[f"{self._generations['B_daughters'][0]}_{self._pdg_var}"] in [
            *self._get_particle_pdg_codes("D**"),
        ]:
            # Only attach D** decay-chain daughters when they are present.
            if (
                len(Xc_daughters) > 0
                and all(
                    self._has_valid_particle(row, prefix)
                    for prefix in self.generations["Xc_grand_daughters"]
                )
            ):
                Xc_grand_daughters = [
                    self._add_hammer_particle(process, row, prefix)
                    for prefix in self.generations["Xc_grand_daughters"]
                ]
                process.add_vertex(Xc_daughters[0], Xc_grand_daughters)

        self.hammer.add_process(process)

        if histogram_name in list(self.histograms.keys()):
            self.hammer.fill_event_histogram(
                histogram_name,
                [row[var] for var in self.histograms[histogram_name]["variables"]],
            )
        self.hammer.process_event()

        if self._get_vertex_decay(row) != "" and verbose:
            logging.info("#################################################")
            logging.info("Processing event: %s", row["__event__"])
            logging.info("Vertex decay %s", self._get_vertex_decay(row))

            for scheme in self.hammer.get_ff_scheme_names():
                try:
                    logging.info("%s , %s", scheme, self.hammer.get_weight(scheme))
                except:
                    logging.info("Probably didn't find eigenvariation")

        return self._collect_results(row)

    def _collect_results(self, row):
        hammer_id_string: str = self._get_vertex_decay(row)
        hammer_scheme_names = self.hammer.get_ff_scheme_names()

        weights = [self.hammer.get_weight(scheme) for scheme in hammer_scheme_names]
        try:
            denom_rate: float = self.rate_container.get_or_calculate_input_rate(
                hammer_id_str=hammer_id_string,
                hammer_instance=self.hammer,
            )
        except OverflowError:
            logging.warning(
                "Event %s threw an OverflowError. No rates were found", row["__event__"]
            )
            denom_rate = -1.0

        try:
            new_rates: List[float] = [
                self.rate_container.get_or_calculate_rate(
                    scheme_name=this_scheme_name,
                    hammer_id_str=hammer_id_string,
                    hammer_instance=self.hammer,
                )
                for this_scheme_name in hammer_scheme_names
            ]
            new_rate: float = new_rates[0]
        except OverflowError:
            logging.warning(
                "Event %s threw an OverflowError. No rates were found", row["__event__"]
            )
            new_rate = -1.0
            new_rates = [new_rate] * len(hammer_scheme_names)

        hammer_reweighted = 0 if (np.array(weights) == 1).all() else 1
        hammer_found_rates = (
            0 if denom_rate in [0.0, 1.0] or new_rate in [0.0, -1.0] else 1
        )

        weights_arr = np.array(weights, dtype=float)
        new_rates_arr = np.array(new_rates, dtype=float)
        scaled_arr = np.ones_like(weights_arr, dtype=float)
        valid = np.isfinite(new_rates_arr) & (new_rates_arr != 0.0)
        scaled_arr[valid] = weights_arr[valid] * denom_rate / new_rates_arr[valid]
        scaled_weights: List[float] = list(scaled_arr)

        return (
            *scaled_weights,
            denom_rate,
            new_rate,
            hammer_reweighted,
            hammer_found_rates,
        )

    def _add_unweighted_results(self):

        weights = [1 for scheme in self.hammer.get_ff_scheme_names()]
        denom_rate, new_rate, hammer_reweighted, hammer_found_rates = 1, 1, -1, -1

        return *weights, denom_rate, new_rate, hammer_reweighted, hammer_found_rates

    def _add_hammer_particle(self, process, row, prefix):

        pdg_value = int(row[f"{prefix}_{self._pdg_var}"])
        return process.add_particle(
            Particle(
                FourMomentum(*[row[var] for var in self._build_column_names(prefix)]),
                pdg_value,
            )
        )

    def _has_valid_particle(self, row, prefix):
        pdg_name = f"{prefix}_{self._pdg_var}"
        if pdg_name not in row.index or not np.isfinite(row[pdg_name]):
            return False

        for var in self._build_column_names(prefix):
            if var not in row.index or not np.isfinite(row[var]):
                return False
        return True

    def _build_column_names(self, prefix):
        return [f"{prefix}_{var}" for var in self._four_momentum_vars]

    def add_histogram(
        self,
        name: str,
        bin_sizes: List[int],
        variables: List[str],
        filename: str,
        ranges: List[Tuple[float, float]] = [],
        has_under_over_flow: bool = True,
    ):

        self.hammer.add_histogram(name, bin_sizes, has_under_over_flow, ranges)
        self._add_histogram_info(name, variables, filename)

    def add_custom_histogram(
        self,
        name: str,
        bin_edges: List[List[float]],
        variables: List[str],
        filename: str,
        has_under_over_flow: bool = True,
    ):

        self.hammer.add_custom_histogram(name, bin_edges, has_under_over_flow)

        self._add_histogram_info(name, variables, filename)

    def _add_histogram_info(self, name, variables, filename):

        self.hammer.keep_errors_in_histogram(name, True)
        self.hammer.collapse_processes_in_histogram(name)

        self.histograms[name] = {}
        self.histograms[name]["variables"] = variables
        self.histograms[name]["filename"] = filename

    def save_histogram(self, histogram_name):

        with open(self.histograms[histogram_name]["filename"], "wb") as fout:

            BUFFER = self.hammer.save_run_header()
            BUFFER.save(fout)

            BUFFER = self.hammer.save_histogram(histogram_name)
            BUFFER.save(fout)

    def _get_vertex_decay(self, row):

        pdg_to_string_dict = {
            511: "B0",
            -511: "B0bar",
            521: "B+",
            -521: "B-",  # B
            411: "D+",
            -411: "D-",
            421: "D0",
            -421: "D0bar",  # D
            413: "D*+",
            -413: "D*-",
            423: "D*0",
            -423: "D*0bar",  # D*
            10411: "D**0*+",
            -10411: "D**0*-",
            10421: "D**0*0",
            -10421: "D**0*0bar",  # D**: D_0*
            10413: "D**1+",
            -10413: "D**1-",
            10423: "D**10",
            -10423: "D**10bar",  # D**: D_1
            20413: "D**1*+",
            -20413: "D**1*-",
            20423: "D**1*0",
            -20423: "D**1*0bar",  # D**: D_1'
            415: "D**2*+",
            -415: "D**2*-",
            425: "D**2*0",
            -425: "D**2*0bar",  # D**: D_2*
            11: "E",
            -11: "E",
            13: "Mu",
            -13: "Mu",
            15: "Tau",
            -15: "Tau",  # Charged leptons
            12: "Nu",
            -12: "Nu",
            14: "Nu",
            -14: "Nu",
            16: "Nu",
            -16: "Nu",
        }

        try:
            decay_string = pdg_to_string_dict[row[f"{self._generations['B']}_mcPDG"]]
            for prefix in self._generations["B_daughters"]:
                decay_string += pdg_to_string_dict[row[f"{prefix}_mcPDG"]]
        except KeyError:
            decay_string = ""

        return decay_string

    @staticmethod
    def _get_particle_pdg_codes(particle_string):

        if particle_string == "D":
            pdg_codes = [411, -411, 421, -421]
        elif particle_string == "D*":
            pdg_codes = [413, -413, 423, -423]
        elif particle_string == "D**":
            pdg_codes = [
                10411,
                -10411,
                10421,
                -10421,
                10413,
                -10413,
                10423,
                -10423,
                20413,
                -20413,
                20423,
                -20423,
                415,
                -415,
                425,
                -425,
            ]
        elif particle_string == "D**0":
            pdg_codes = [10411, -10411, 10421, -10421]
        elif particle_string == "D**1":
            pdg_codes = [10413, -10413, 10423, -10423]
        elif particle_string == "D**1*":
            pdg_codes = [20413, -20413, 20423, -20423]
        elif particle_string == "D**2*":
            pdg_codes = [415, -415, 425, -425]
        elif particle_string == "tau":
            pdg_codes = [15, -15]
        elif particle_string == "ell":
            pdg_codes = [11, -11, 13, -13]
        elif particle_string == "leptons":
            pdg_codes = [11, -11, 13, -13, 15, -15]
        else:
            raise ValueError("Wrong kind of Xc string was passed")

        return pdg_codes


def get_varied_FF_central_values(model):
    """
    Inherited from Henrik.
    Function to calculate the different +-1sigma variations of the form factors.
    For that, the form factor parameters are rotated into a paramter space in which all of
    them are perfectly orthogonal and uncorrelated (the eigenvectors of the covariance matrix).
    Then, the 1 sigma up and down variation of each of these new parameters is taken (symbolized
    by the eigenvalues) and this is rotated back into the original parameter space.
    One gets 2*N different sets of varied central values (for N parameters of the FF model), each
    presenting the 1sigma up- and down variation of one of the orthogonal parameters.
    Subsequently, these new varied sets can be used to derive the 1 sigma varied histogram with
    eFFORT or HAMMER.
    :param ff_dict: Dictionary that includes the form factor model parameters and their correlations.
                    The parameters need to be of listed with param_{i} and uncert_{i}.
                    Moreover the dict needs to contain the "corrmat" entry which includes the
                    correlation matrix with the correct paramter order and the "num_param" entry
                    which specifies the number of parameters in this form factor model.
                    If statistical and systematic errors are distinguished, the dict needs entries
                    "stat_{i}", "syst_{i}" and corrmat_stat / _syst.
    """
    cvs = np.array(
        [
            param.nominal_value if name != "DelMbc" else param["DelMbc"].nominal_value
            for name, param in model.params.items()
        ]
    )
    errors = np.array(
        [
            param.std_dev if name != "DelMbc" else param["DelMbc"].std_dev
            for name, param in model.params.items()
        ]
    )
    covariance_matrix = corr2cov(model.corr_matrix, errors)

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # If DelMbc is one of the parameters, the charm mass mc must be derived from
    # it (mc = mb - DelMbc) to still populate the HAMMER parameter string below.
    param_names = [name for name in model.params.keys()]
    if "DelMbc" in param_names:
        logging.warning(
            "WARNING! The selected form factor model contains mb and mc. Please take extra care that everything works as expected."
        )
        mb_index = param_names.index("mb")
        DelMbc_index = param_names.index("DelMbc")

    eigenvalue_variations = []
    for i, eigenvalue in enumerate(eigenvalues):
        variation = np.zeros(len(cvs))
        variation[i] = np.sqrt(eigenvalue)

        up = cvs + eigenvectors @ variation
        down = cvs - eigenvectors @ variation

        if "DelMbc" in param_names:
            logging.info(
                f"Calculate mc for up (mc = {up[mb_index]:.3f} - {up[DelMbc_index]:.3f}) and down ({down[mb_index]:.3f} - {down[DelMbc_index]:.3f}) variations."
            )
            up[DelMbc_index] = up[mb_index] - up[DelMbc_index]
            down[DelMbc_index] = down[mb_index] - down[DelMbc_index]
            logging.info(
                f"Results: up: mc = {up[DelMbc_index]:.3f}, down: mc = {down[DelMbc_index]:.3f}."
            )

        eigenvalue_variations.append((up, down))

    # in the case of the broad D**, additionally add the 1.5 sigma variations to account for wrong modelling
    # due to the non-consideration of their sizable widths in the theory prediction and in HAMMER:
    if model.Xc in ["D**0*", "D**1*"]:
        for i, eigenvalue in enumerate(eigenvalues):
            variation = np.zeros(len(cvs))
            variation[i] = np.sqrt(eigenvalue) * 1.5

            up = cvs + eigenvectors @ variation
            down = cvs - eigenvectors @ variation

            eigenvalue_variations.append((up, down))

    return eigenvalue_variations
