"""Utilities and definitions for the follicle ellongation model

"""
import logging

import numpy as np
import matplotlib.pyplot as plt

import ipyvolume as ipv
import ipywidgets as widgets
from IPython.display import display

from tyssue import HistoryHdf5, EventManager, Monolayer

from tyssue.draw import sheet_view
from tyssue.geometry.bulk_geometry import ClosedMonolayerGeometry as geom
from tyssue.io import hdf5

from tyssue.dynamics import model_factory, effectors
from tyssue.dynamics import units
from tyssue.dynamics.sheet_gradients import area_grad

from tyssue.generation.shapes import spherical_monolayer

from tyssue.behaviors import increase, reconnect
from tyssue.solvers.viscous import EulerSolver
from tyssue.solvers.quasistatic import QSSolver

from tyssue.utils.decorators import cell_lookup
from tyssue.topology.monolayer_topology import cell_division
from tyssue.topology.bulk_topology import fix_pinch


from tyssue.utils import to_nd

logger = logging.getLogger("tyssue")
logger.setLevel("DEBUG")


class WAMonolayerGeometry(geom):
    """Monolayer geometry class that adds a 'weighted  area' term
    to the faces of the epithelium.

    At each update, the weight on the faces is normalized so that
    the sum of face weights for a cell is equal to its number of faces.

    """

    @classmethod
    def update_all(cls, eptm):

        super().update_all(eptm)
        cls.normalize_face_weights(eptm)
        cls.update_weithed_area(eptm)

    @staticmethod
    def normalize_face_weights(eptm):

        eptm.face_df["cell"] = eptm.edge_df.groupby("face")["cell"].first()
        sum_weights = eptm.face_df.groupby("cell")["weight"].sum()
        weight_norm = (eptm.cell_df["num_faces"] / sum_weights).loc[
            eptm.face_df["cell"].to_numpy()
        ]
        eptm.face_df["weight"] *= weight_norm.to_numpy()

    @staticmethod
    def update_weithed_area(eptm):
        eptm.edge_df["face_weight"] = eptm.upcast_face(eptm.face_df["weight"])
        eptm.cell_df["weighted_area"] = eptm.sum_cell(
            eptm.edge_df.eval("sub_area * face_weight")
        )


class WeightedCellAreaElasticity(effectors.AbstractEffector):
    """Cell area elasticity with weighted areas

    .. math::

        E_{wa} = \sum_c \frac{K_A}{2} (A'_c - A_0)^{1/2}
        A'_c = \sum_{\alpha \in c} w_\alpha A_\alpha

    """

    dimensions = units.area_elasticity
    magnitude = "area_elasticity"
    label = "Weighted Area elasticity"
    element = "cell"
    specs = {
        "cell": {
            "is_alive": 1,
            "weighted_area": 1.0,
            "area_elasticity": 1.0,
            "prefered_area": 1.0,
        },
        "face": {"area": 1.0, "weight": 1.0},
    }
    spatial_ref = "prefered_area", units.area

    @staticmethod
    def get_nrj_norm(specs):
        return specs["cell"]["area_elasticity"] * specs["cell"]["prefered_area"] ** 2

    @staticmethod
    def energy(eptm):

        return effectors.elastic_energy(
            eptm.cell_df, "weighted_area", "area_elasticity * is_alive", "prefered_area"
        )

    @staticmethod
    def gradient(eptm):
        ka_a0_ = effectors.elastic_force(
            eptm.cell_df, "weighted_area", "area_elasticity * is_alive", "prefered_area"
        )
        face_weight = eptm.edge_df["face_weight"].to_numpy()

        ka_a0 = to_nd(eptm.upcast_cell(ka_a0_) * face_weight, 3)

        grad_a_srce, grad_a_trgt = area_grad(eptm)

        grad_a_srce = ka_a0 * grad_a_srce
        grad_a_trgt = ka_a0 * grad_a_trgt
        grad_a_srce.columns = ["g" + u for u in eptm.coords]
        grad_a_trgt.columns = ["g" + u for u in eptm.coords]

        return grad_a_srce, grad_a_trgt


def get_initial_follicle(specs, recreate=False, Nc=200):
    """Retrieves or recreates a spherical epithelium"""
    if recreate:
        ## Lloy_relax=True takes time but should give more spherical epithelium
        follicle = spherical_monolayer(9.0, 12.0, Nc, apical="in", Lloyd_relax=True)
        geom.update_all(follicle)
        geom.scale(follicle, follicle.cell_df.vol.mean() ** (-1 / 3), list("xyz"))
        geom.update_all(follicle)
    else:
        follicle = Monolayer("follicle", hdf5.load_datasets("initial_follicle.hf5"))
        follicle.settings["lumen_side"] = "apical"
        geom.update_all(follicle)

    follicle.update_specs(specs, reset=True)
    follicle.cell_df["id"] = follicle.cell_df.index

    wgeom = WAMonolayerGeometry
    model = model_factory(
        [
            effectors.LumenVolumeElasticity,
            WeightedCellAreaElasticity,
            effectors.CellVolumeElasticity,
        ]
    )

    print("Finding static equilibrium")

    follicle.face_df.loc[follicle.apical_faces, "weight"] = specs["settings"][
        "apical_weight"
    ]
    wgeom.update_all(follicle)

    solver = QSSolver()
    res = solver.find_energy_min(follicle, wgeom, model)

    return follicle, model


def contractility_grad(follicle, cell, amp, span, coords=["x", "y", "z"], elem="face"):
    """Returns values exponentialy decreasing as a function of the (euclidian) distance
    from the reference cell.

    `grad = amp * np.exp(-distance / span)

    The values are evaluated for the element passed as argument.

    Parameters
    ----------

    follicle : a Monolayer
    cell : int
        the index of the reference cell
    amp : float
        the amplitude of the gradient
    span : float
        the gradient width at 1/e
    coords : column index, default ['x', 'y', 'z']
    elem : str {'cell'|'face'|'vert'}
        element over which to evaluate de gradient
    """
    cell_pos = follicle.cell_df.loc[cell, coords].to_numpy()
    distance = np.linalg.norm(
        follicle.datasets[elem][coords] - cell_pos[None, :], axis=1
    )

    grad = amp * np.exp(-distance / span)
    return grad  # - grad.min()


def get_polar_cells(follicle, ids=None):
    """Returns the index of the anterior-most and posterior-most cells

    If ids are passed, returns the index of the cells with those ids,
    else, returns the indices of the two cells with minimal and maximal z

    """

    if ids is None:
        ante_cell = follicle.cell_df["z"].idxmin()
        post_cell = follicle.cell_df["z"].idxmax()
    else:
        ante_cell, post_cell = follicle.cell_df[follicle.cell_df["id"].isin(ids)].index

    return ante_cell, post_cell


def update_gradient(follicle, polar_cells, amp, span):
    """Updates `follicle.face_df` "contractile_grad" column by evaluating
    the gradient from both polar cells, then updates the face "weights"

    """

    ante_cell, post_cell = polar_cells

    ante_faces = follicle.edge_df.query(f"cell == {ante_cell}")["face"].unique()
    post_faces = follicle.edge_df.query(f"cell == {post_cell}")["face"].unique()
    ante_grad = contractility_grad(follicle, ante_cell, amp=amp, span=span)
    post_grad = contractility_grad(follicle, post_cell, amp=amp, span=span)

    follicle.face_df["contractile_grad"] = ante_grad + post_grad
    min_grad = follicle.face_df["contractile_grad"].min()

    follicle.face_df.loc[ante_faces, "contractile_grad"] = min_grad
    follicle.face_df.loc[post_faces, "contractile_grad"] = min_grad

    follicle.face_df["weight"] = 1.0
    follicle.face_df.loc[follicle.apical_faces, "weight"] += follicle.face_df.loc[
        follicle.apical_faces, "contractile_grad"
    ]
    WAMonolayerGeometry.update_all(follicle)


class MonolayerView(widgets.HBox):
    def __init__(self, eptm, **draw_specs):

        plt.ioff()
        ipv.clear()
        self.fig3D, self.mesh = sheet_view(eptm, mode="3D", **draw_specs)
        self.graph_widget = widgets.Output()
        with self.graph_widget:

            self.fig2D, (self.ax0, self.ax1) = plt.subplots(
                2, 1, sharey=True, sharex=True
            )

            apical = eptm.get_sub_sheet("apical")
            apical.reset_index()
            apical.reset_topo()
            apical.face_df["visible"] = apical.face_df["y"] > 0
            _ = sheet_view(
                apical,
                mode="2D",
                coords=["z", "x"],
                ax=self.ax0,
                edge={"visible": False},
                face={"visible": True, "color": apical.face_df.area},
            )
            basal = eptm.get_sub_sheet("basal")
            basal.reset_index()
            basal.reset_topo()
            basal.face_df["visible"] = basal.face_df["y"] > 0
            _ = sheet_view(
                basal,
                mode="2D",
                coords=["z", "x"],
                ax=self.ax1,
                edge={"visible": False},
                face={"visible": True, "color": basal.face_df.area},
            )
            self.ax0.set_title("Apical mesh")
            self.ax1.set_title("Basal mesh")
            self.ax0.set_axis_off()
            self.ax1.set_axis_off()
            self.fig2D.set_size_inches(5, 8)
            plt.close(self.fig2D)
            display(self.fig2D)

        super().__init__([self.fig3D, self.graph_widget])


def reset_segments(monolayer):
    try:
        monolayer.get_opposite_faces()
    except ValueError:
        fix_pinch(monolayer)

    face_normals = monolayer.face_df[monolayer.coords].copy()
    for nu in monolayer.ncoords:
        face_normals[nu[1]] = monolayer.sum_face(monolayer.edge_df[nu])

    proj = (face_normals * monolayer.face_df[monolayer.coords].to_numpy()).sum(axis=1)

    monolayer.face_df.loc[proj > 0, "segment"] = "basal"
    monolayer.face_df.loc[proj < 0, "segment"] = "apical"
    monolayer.face_df.loc[monolayer.face_df.opposite != -1, "segment"] = "lateral"
    monolayer.face_df["visible"] = monolayer.face_df["segment"] == "apical"
    monolayer.edge_df["segment"] = monolayer.upcast_face("segment")


default_division_spec = {
    "cell": -1,
    "growth_rate": 0.02,
    "growth_noise": 0.02,
    "critical_vol": 2.0,
    "autonomous": True,
}

# This will go in tyssue in the next release
@cell_lookup
def division(mono, manager, **kwargs):
    """Cell division happens through cell growth up to a critical volume,
    followed by actual division of the cell.

    Parameters
    ----------
    mono : a `Monolayer` instance
    manager : an `EventManager` instance
    cell_id : int,
      index of the mother cell
    growth_rate : float, default 0.1
      rate of increase of the prefered volume
    critical_vol : float, default 2.
      volume at which the cells stops to grow and devides
    """
    division_spec = default_division_spec
    division_spec.update(**kwargs)

    cell = division_spec["cell"]
    Vc = division_spec["critical_vol"] * mono.specs["cell"]["prefered_vol"]

    if mono.cell_df.loc[cell, "vol"] < Vc:

        growth_rate = np.random.normal(
            loc=division_spec["growth_rate"], scale=division_spec["growth_noise"]
        )
        dv = 1 + growth_rate * mono.settings["dt"]
        increase(mono, "cell", cell, dv, "prefered_vol")
        manager.append(division, **division_spec)
    else:
        mono.cell_df.loc[cell, "prefered_vol"] = mono.specs["cell"]["prefered_vol"]
        mono.cell_df.loc[cell, "prefered_area"] = mono.specs["cell"]["prefered_area"]
        logger.info(f"{manager.clock:.2f}: division of cell {cell}")
        orientation = division_spec.get("orientation", "apical")
        daughter = cell_division(mono, cell, "vertical")
        if division_spec["autonomous"]:
            manager.append(division, **division_spec)
            try:
                division_spec["cell_id"] = mono.cell_df.loc[daughter, "id"]
            except KeyError:
                logger.error("cell %d not found" % daughter)
        reset_segments(mono)


def set_gradient(mono, manager, **kwargs):

    polar_cells = get_polar_cells(mono, kwargs.get("polar_cells_ids"))
    update_gradient(mono, polar_cells, kwargs.get("amp", 1.0), kwargs.get("span", 1.0))
    manager.append(set_gradient, **kwargs)


def lumen_growth(mono, manager, **kwargs):

    growth_rate = kwargs.get("lumen_growth_rate", 0.014)
    dt = mono.settings["dt"]
    mono.settings["lumen_prefered_vol"] *= 1 + growth_rate * dt
    manager.append(lumen_growth, **kwargs)


def check_opposite(mono, manager, **kwargs):

    mono.get_opposite_faces()
    manager.append(check_opposite, **kwargs)


def get_solver(follicle, model, dt, base_dir, history_file, parameters, save_interval):

    eptm = follicle.copy()
    eptm.edge_df[["srce", "trgt"]] = eptm.edge_df[["srce", "trgt"]].astype(int)
    eptm.settings["dt"] = dt

    manager = EventManager("cell")
    # Auto solve rearangements
    manager.append(reconnect)
    manager.append(set_gradient, **parameters)

    for cell_id in eptm.cell_df["id"]:
        if cell_id in polar_cells:
            continue
        manager.append(division, cell_id=cell_id, **parameters)

    manager.append(lumen_growth, **parameters)
    manager.append(check_opposite)

    history = HistoryHdf5(
        eptm, save_every=save_interval, dt=dt, hf5file=base_dir / history_file
    )

    solver = EulerSolver(
        eptm,
        geom,
        model,
        manager=manager,
        history=history,
        bounds=(-eptm.edge_df.length.median(), eptm.edge_df.length.median()),
    )

    manager.update()
    return solver
