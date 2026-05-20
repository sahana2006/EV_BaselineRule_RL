from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from xml.dom import minidom
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = Path(__file__).resolve().parent
NET_PATH = SCENARIO_DIR / "grid.net.xml"
ROUTES_PATH = SCENARIO_DIR / "grid_routes.rou.xml"
SUMOCFG_PATH = SCENARIO_DIR / "grid.sumocfg"
VIEW_SETTINGS_PATH = SCENARIO_DIR / "grid_view.settings.xml"


def ensure_sumo_tools() -> tuple[str, str]:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise EnvironmentError("SUMO_HOME is not set. Please set SUMO_HOME before generating the grid.")

    tools_dir = Path(sumo_home) / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.append(str(tools_dir))

    bin_dir = Path(sumo_home) / "bin"
    netgenerate = bin_dir / "netgenerate.exe"
    if not netgenerate.is_file():
        netgenerate = bin_dir / "netgenerate"

    if not netgenerate.is_file():
        raise FileNotFoundError(f"netgenerate was not found under {bin_dir}")

    return sumo_home, str(netgenerate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reusable 4x4 SUMO traffic grid scenario.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic SUMO seed written to grid.sumocfg")
    parser.add_argument("--duration", type=int, default=3600, help="Simulation end time in seconds")
    parser.add_argument(
        "--density",
        type=float,
        default=1.0,
        help="Traffic scaling factor applied to base civilian veh/hr per entry edge",
    )
    parser.add_argument("--spacing", type=float, default=200.0, help="Intersection spacing in meters")
    parser.add_argument("--lanes", type=int, default=2, help="Lanes per direction")
    parser.add_argument("--grid-size", type=int, default=4, help="Number of traffic-light intersections per axis")
    parser.add_argument(
        "--base-vph",
        type=int,
        default=240,
        help="Base civilian vehicles per hour generated for each inbound boundary edge",
    )
    return parser.parse_args()


def run_command(command: Sequence[str]) -> None:
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def generate_network(netgenerate_binary: str, args: argparse.Namespace) -> None:
    command = [
        netgenerate_binary,
        "--grid",
        "--grid.x-number",
        str(args.grid_size),
        "--grid.y-number",
        str(args.grid_size),
        "--grid.x-length",
        str(args.spacing),
        "--grid.y-length",
        str(args.spacing),
        "--default-junction-type",
        "traffic_light",
        "--default.lanenumber",
        str(args.lanes),
        "--tls.green.time",
        "31",
        "--tls.yellow.time",
        "4",
        "--tls.allred.time",
        "2",
        "--no-turnarounds.tls",
        "--output-file",
        str(NET_PATH),
    ]
    run_command(command)


@dataclass(frozen=True)
class BoundaryEdges:
    west_in: list
    east_in: list
    south_in: list
    north_in: list
    west_out: list
    east_out: list
    south_out: list
    north_out: list


def classify_boundary_edges(net) -> BoundaryEdges:
    edges = [edge for edge in net.getEdges() if not edge.getID().startswith(":")]
    nodes = [node for node in net.getNodes() if not node.getID().startswith(":")]
    xs = [node.getCoord()[0] for node in nodes]
    ys = [node.getCoord()[1] for node in nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    tol = 1e-6

    def direction(edge) -> str:
        from_x, from_y = edge.getFromNode().getCoord()
        to_x, to_y = edge.getToNode().getCoord()
        dx = to_x - from_x
        dy = to_y - from_y
        if abs(dx) >= abs(dy):
            return "E" if dx > 0 else "W"
        return "N" if dy > 0 else "S"

    west_in = sorted(
        [edge for edge in edges if direction(edge) == "E" and abs(edge.getFromNode().getCoord()[0] - min_x) < tol],
        key=lambda edge: edge.getFromNode().getCoord()[1],
    )
    east_in = sorted(
        [edge for edge in edges if direction(edge) == "W" and abs(edge.getFromNode().getCoord()[0] - max_x) < tol],
        key=lambda edge: edge.getFromNode().getCoord()[1],
    )
    south_in = sorted(
        [edge for edge in edges if direction(edge) == "N" and abs(edge.getFromNode().getCoord()[1] - min_y) < tol],
        key=lambda edge: edge.getFromNode().getCoord()[0],
    )
    north_in = sorted(
        [edge for edge in edges if direction(edge) == "S" and abs(edge.getFromNode().getCoord()[1] - max_y) < tol],
        key=lambda edge: edge.getFromNode().getCoord()[0],
    )

    west_out = sorted(
        [edge for edge in edges if direction(edge) == "W" and abs(edge.getToNode().getCoord()[0] - min_x) < tol],
        key=lambda edge: edge.getToNode().getCoord()[1],
    )
    east_out = sorted(
        [edge for edge in edges if direction(edge) == "E" and abs(edge.getToNode().getCoord()[0] - max_x) < tol],
        key=lambda edge: edge.getToNode().getCoord()[1],
    )
    south_out = sorted(
        [edge for edge in edges if direction(edge) == "S" and abs(edge.getToNode().getCoord()[1] - min_y) < tol],
        key=lambda edge: edge.getToNode().getCoord()[0],
    )
    north_out = sorted(
        [edge for edge in edges if direction(edge) == "N" and abs(edge.getToNode().getCoord()[1] - max_y) < tol],
        key=lambda edge: edge.getToNode().getCoord()[0],
    )
    return BoundaryEdges(west_in, east_in, south_in, north_in, west_out, east_out, south_out, north_out)


def route_edge_ids(net, origin, destination) -> list[str]:
    route, _cost = net.getShortestPath(origin, destination)
    if not route:
        raise RuntimeError(f"No route found between {origin.getID()} and {destination.getID()}")
    return [edge.getID() for edge in route if not edge.getID().startswith(":")]


def append_route(parent: ET.Element, route_id: str, edge_ids: Iterable[str]) -> None:
    ET.SubElement(parent, "route", id=route_id, edges=" ".join(edge_ids))


def add_distribution(
    parent: ET.Element,
    distribution_id: str,
    route_ids: Sequence[str],
    probabilities: Sequence[float],
) -> None:
    distribution = ET.SubElement(parent, "routeDistribution", id=distribution_id)
    for route_id, probability in zip(route_ids, probabilities):
        ET.SubElement(distribution, "route", refId=route_id, probability=f"{probability:.2f}")


def build_civilian_routes(net, routes_root: ET.Element, boundary: BoundaryEdges) -> None:
    route_ids_seen: set[str] = set()

    def route_name(prefix: str, source_index: int, variant: str) -> str:
        return f"{prefix}_{source_index}_{variant}"

    def maybe_add_route(route_id: str, edge_ids: list[str]) -> None:
        if route_id in route_ids_seen:
            return
        append_route(routes_root, route_id, edge_ids)
        route_ids_seen.add(route_id)

    def build_family(
        prefix: str,
        sources: Sequence,
        straight_targets: Sequence,
        turn_a_targets: Sequence,
        turn_b_targets: Sequence,
        turn_a_shift: int,
        turn_b_shift: int,
    ) -> None:
        for idx, source in enumerate(sources):
            straight_target = straight_targets[idx]
            turn_a_target = turn_a_targets[min(len(turn_a_targets) - 1, idx + turn_a_shift)]
            turn_b_target = turn_b_targets[max(0, idx - turn_b_shift)]

            route_ids = [
                route_name(prefix, idx, "straight"),
                route_name(prefix, idx, "turn_a"),
                route_name(prefix, idx, "turn_b"),
            ]
            maybe_add_route(route_ids[0], route_edge_ids(net, source, straight_target))
            maybe_add_route(route_ids[1], route_edge_ids(net, source, turn_a_target))
            maybe_add_route(route_ids[2], route_edge_ids(net, source, turn_b_target))

            add_distribution(
                routes_root,
                f"{prefix}_dist_{idx}",
                route_ids,
                probabilities=(0.50, 0.25, 0.25),
            )
            ET.SubElement(
                routes_root,
                "flow",
                id=f"{prefix}_flow_{idx}",
                type="civilian",
                route=f"{prefix}_dist_{idx}",
                begin="0",
                end="$DURATION$",
                vehsPerHour="$VPH$",
                departLane="best",
                departSpeed="max",
            )

    build_family(
        prefix="west_in",
        sources=boundary.west_in,
        straight_targets=boundary.east_out,
        turn_a_targets=boundary.north_out,
        turn_b_targets=boundary.south_out,
        turn_a_shift=1,
        turn_b_shift=0,
    )
    build_family(
        prefix="east_in",
        sources=boundary.east_in,
        straight_targets=boundary.west_out,
        turn_a_targets=boundary.south_out,
        turn_b_targets=boundary.north_out,
        turn_a_shift=1,
        turn_b_shift=0,
    )
    build_family(
        prefix="south_in",
        sources=boundary.south_in,
        straight_targets=boundary.north_out,
        turn_a_targets=boundary.east_out,
        turn_b_targets=boundary.west_out,
        turn_a_shift=1,
        turn_b_shift=0,
    )
    build_family(
        prefix="north_in",
        sources=boundary.north_in,
        straight_targets=boundary.south_out,
        turn_a_targets=boundary.west_out,
        turn_b_targets=boundary.east_out,
        turn_a_shift=1,
        turn_b_shift=0,
    )


def build_emergency_routes(net, routes_root: ET.Element, boundary: BoundaryEdges) -> None:
    ev_specs = [
        (
            "ev_route_horizontal",
            "ev_0",
            route_edge_ids(net, boundary.west_in[1], boundary.east_out[1]),
            "20",
        ),
        (
            "ev_route_vertical",
            "ev_1",
            route_edge_ids(net, boundary.south_in[2], boundary.north_out[2]),
            "80",
        ),
        (
            "ev_route_mixed",
            "ev_2",
            route_edge_ids(net, boundary.west_in[0], boundary.north_out[-1]),
            "140",
        ),
    ]

    for route_id, vehicle_id, edge_ids, depart in ev_specs:
        append_route(routes_root, route_id, edge_ids)
        ET.SubElement(
            routes_root,
            "vehicle",
            id=vehicle_id,
            type="emergency",
            route=route_id,
            depart=depart,
            departLane="best",
            departSpeed="max",
        )


def write_routes_file(args: argparse.Namespace) -> None:
    from sumolib.net import readNet  # type: ignore

    net = readNet(str(NET_PATH))
    boundary = classify_boundary_edges(net)

    routes_root = ET.Element(
        "routes",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
        },
    )
    ET.SubElement(
        routes_root,
        "vType",
        id="civilian",
        vClass="passenger",
        accel="2.6",
        decel="4.5",
        sigma="0.4",
        length="5.0",
        minGap="2.5",
        maxSpeed="13.9",
        color="0,90,255",
    )
    ET.SubElement(
        routes_root,
        "vType",
        id="emergency",
        vClass="emergency",
        guiShape="emergency",
        accel="4.0",
        decel="5.0",
        sigma="0.1",
        length="6.5",
        minGap="1.0",
        maxSpeed="22.0",
        speedFactor="1.25",
        color="255,0,0",
    )

    build_civilian_routes(net, routes_root, boundary)
    build_emergency_routes(net, routes_root, boundary)

    xml_bytes = ET.tostring(routes_root, encoding="utf-8")
    pretty_xml = minidom.parseString(xml_bytes).toprettyxml(indent="    ")
    vehicles_per_hour = max(60, int(round(args.base_vph * args.density)))
    pretty_xml = pretty_xml.replace("$DURATION$", str(args.duration)).replace("$VPH$", str(vehicles_per_hour))
    ROUTES_PATH.write_text(pretty_xml, encoding="utf-8")


def write_sumocfg(args: argparse.Namespace) -> None:
    config = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="grid.net.xml"/>
        <route-files value="grid_routes.rou.xml"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="{args.duration}"/>
        <step-length value="1"/>
    </time>

    <random_number>
        <seed value="{args.seed}"/>
    </random_number>

    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
    </report>

    <gui_only>
        <gui-settings-file value="grid_view.settings.xml"/>
    </gui_only>
</configuration>
"""
    SUMOCFG_PATH.write_text(config, encoding="utf-8")


def write_view_settings(args: argparse.Namespace) -> None:
    span = args.spacing * max(1, args.grid_size - 1)
    center = span / 2.0
    zoom = max(30, int(args.spacing * 0.42))
    view_settings = f"""<?xml version="1.0" encoding="UTF-8"?>
<viewsettings>
    <viewport zoom="{zoom}" x="{center:.1f}" y="{center:.1f}"/>
    <scheme name="real world"/>
    <delay value="20"/>
</viewsettings>
"""
    VIEW_SETTINGS_PATH.write_text(view_settings, encoding="utf-8")


def main() -> None:
    args = parse_args()
    _sumo_home, netgenerate_binary = ensure_sumo_tools()
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)

    generate_network(netgenerate_binary, args)
    write_routes_file(args)
    write_sumocfg(args)
    write_view_settings(args)

    print("Generated grid scenario assets:")
    print(f"  - {NET_PATH}")
    print(f"  - {ROUTES_PATH}")
    print(f"  - {SUMOCFG_PATH}")
    print(f"  - {VIEW_SETTINGS_PATH}")
    print(f"Traffic density: {args.density} (base {args.base_vph} veh/hr per entry edge)")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
