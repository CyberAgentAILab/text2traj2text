"""PRM (Probabilistic Roadmap) global planner.

Author: Hikaru Asano
Affiliation: CyberAgent AI Lab / The University of Tokyo
"""

from __future__ import annotations

from logging import getLogger
from typing import NamedTuple

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from chex import Array, PRNGKey, dataclass
from jax.tree_util import tree_map

from ..core.core import Position
from .core import Layout

logger = getLogger(__name__)


class Roadmap(NamedTuple):
    G: nx.Graph
    samples: Position


@dataclass
class PRM:
    layout: Layout
    roadmap: Roadmap = None
    num_samples: int = 100
    seed: int = 0
    max_iter: int = 10

    def __post_init__(self):
        self.draw_samples = self._build_draw_samples()
        self.compute_edge_data = self._build_compute_edge_data()
        self.construct_roadmap(jax.random.PRNGKey(self.seed), self.num_samples)

    def _build_draw_samples(self):
        def draw_sample(key: PRNGKey) -> Position:
            class WhileCarry(NamedTuple):
                sample: Position
                key: PRNGKey
                num_iter: int

            def while_cond(carry: WhileCarry):
                cond1 = self.layout.validate_sample(carry.sample)
                cond2 = carry.num_iter < self.max_iter
                cond3 = jnp.any(jnp.isinf(carry.sample.array()))

                return (~cond1 & cond2) | cond3

            def while_body(carry: WhileCarry):
                new_key, key = jax.random.split(carry.key)
                x, y = jax.random.uniform(new_key, (2,))
                sample = Position(x, y)

                return carry._replace(
                    key=key,
                    sample=sample,
                    num_iter=carry.num_iter + 1,
                )

            while_carry = jax.lax.while_loop(
                while_cond, while_body, WhileCarry(Position(jnp.inf, jnp.inf), key, 0)
            )

            return while_carry

        def draw_samples(key: PRNGKey, num_samples: int) -> Position:
            """
            Draw the specified number of samples

            Args:
                key (PRNGKey): PRNG key
                num_samples (int): number of samples

            Returns:
                Position: valid samples
            """

            class LoopCarry(NamedTuple):
                samples: Position
                key: PRNGKey

            def loop_body(i: int, carry: LoopCarry):
                while_carry = draw_sample(carry.key)
                xs = carry.samples.x.at[i].set(while_carry.sample.x)
                ys = carry.samples.y.at[i].set(while_carry.sample.y)
                return carry._replace(
                    key=while_carry.key,
                    samples=Position(xs, ys),
                )

            xs = jnp.ones((num_samples,)) * jnp.inf
            ys = jnp.ones((num_samples,)) * jnp.inf
            samples = jax.lax.fori_loop(
                0,
                num_samples,
                loop_body,
                LoopCarry(Position(xs, ys), key),
            ).samples

            return samples

        return jax.jit(draw_samples, static_argnames={"num_samples"})

    def _build_compute_edge_data(self):
        def compute_edge_data(
            samples: Position, new_sample: Position, new_sample_id: int
        ) -> Array:
            """
            Compute edge data for a new sample and already existing samples

            Args:
                samples (Position): samples already drawn
                new_sample (Position): new sample
                new_sample_id (int): new sample id to be added

            Returns:
                Array: array of (src_id, new_sample_id, edge length, edge validity)
            """
            edges = jax.vmap(self.layout.validate_line, in_axes=(0, None))(
                samples, new_sample
            )
            edge_dist = jnp.linalg.norm((samples - new_sample).array(), axis=-1)
            src_id = jnp.arange(len(samples))

            edge_data = jax.vmap(
                lambda si, e, ed: jnp.hstack((si, new_sample_id, ed[si], e[si])),
                in_axes=(0, None, None),
            )(src_id, edges, edge_dist)

            return edge_data

        return jax.jit(compute_edge_data)

    def construct_roadmap(self, key: PRNGKey, num_samples: int) -> None:
        """
        Construct roadmap

        Args:
            key (PRNGKey): PRNG key
            num_samples (int): number of samples for the roadmap
        """

        G = nx.Graph()
        samples = self.draw_samples(key, num_samples)
        edge_data = np.asarray(
            jax.vmap(self.compute_edge_data, in_axes=(None, 0, 0))(
                samples, samples, jnp.arange(len(samples))
            ).reshape(-1, 4)
        )
        G.add_weighted_edges_from(edge_data[edge_data[:, -1] == 1, :-1])

        self.roadmap = Roadmap(G=G, samples=samples)
        self.e = edge_data[edge_data[:, -1] == 1, :-2]
        self.sample = samples

    def update_roadmap(self, position: Position) -> int:
        """
        Update roadmap with a new sample

        Args:
            new_sample (Position): new sample

        Returns:
            int: id associated with the new sample in roadmap.G
        """

        new_sample = position

        roadmap = self.roadmap
        exist = np.all(np.isclose(roadmap.samples.array(), new_sample.array()), axis=-1)
        if np.any(exist):
            # logger.warning(f"new_sample {new_sample} already exists")
            return np.argwhere(exist)[0][0]

        new_sample_id = len(roadmap.samples)
        edge_data = np.asarray(
            self.compute_edge_data(roadmap.samples, new_sample, new_sample_id)
        )
        G, samples = roadmap.G, roadmap.samples
        G.add_node(new_sample_id)
        G.add_weighted_edges_from(edge_data[edge_data[:, -1] == 1, :-1])
        samples = Position(
            np.hstack([samples.x, new_sample.x]), np.hstack([samples.y, new_sample.y])
        )
        self.roadmap = Roadmap(G=G, samples=samples)

        return new_sample_id

    def plan(self, src: Position, dst: Position) -> Position:
        """
        Find shortest path from src to dst

        Args:
            src (Position): src position
            dst (Position): dst position

        Returns:
            Position: shortest path
        """

        src_id = self.update_roadmap(src)
        dst_id = self.update_roadmap(dst)
        path_ids = nx.shortest_path(self.roadmap.G, src_id, dst_id, weight="weight")

        path = [
            Position(
                self.roadmap.samples.at(int(path_id)).x,
                self.roadmap.samples.at(int(path_id)).y,
            )
            for path_id in path_ids
        ]
        path = tree_map(lambda *values: jnp.stack(values), *path)

        return path
