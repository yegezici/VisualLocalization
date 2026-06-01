#!/usr/bin/env python

"""Spawn additional building actors into the current CARLA map.

The spawned buildings are runtime actors. They are visible in the current
simulation session, but they are not baked into the .umap package.
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

carla = None


def load_carla_module():
    global carla
    if carla is not None:
        return carla

    try:
        platform_tag = 'win-amd64' if os.name == 'nt' else 'linux-x86_64'
        dist_dir = Path(__file__).resolve().parents[1] / 'carla' / 'dist'
        egg_patterns = [
            str(dist_dir / 'carla-*%d.%d-%s.egg') % (
                sys.version_info.major,
                sys.version_info.minor,
                platform_tag),
            str(dist_dir / 'carla-*py3.*-%s.egg') % platform_tag,
        ]
        sys.path.append(next(path for pattern in egg_patterns for path in glob.glob(pattern)))
    except StopIteration:
        pass

    import carla as carla_module
    carla = carla_module
    return carla


DEFAULT_BUILDINGS = [
    {
        "id": "marmara_default_01",
        "type": "static.prop.streetbarrier",
        "location": {"x": 151.61314392089844, "y": 108.13982391357422, "z":-9.624250411987305},
        "rotation": {"pitch": 0.0, "yaw": 0, "roll": 0.0},
        "size": {"x": 12.0, "y": 10.0, "z": 18.0},
        "snap_to_ground": True
    },
    {
        "id": "marmara_default_02",
        "type": "static.prop.streetbarrier",
        "location": {"x": 160.8955078125, "y": 106.52881622314453, "z": -9.81245231628418},
        "rotation": {"pitch": 0.0, "yaw": -20, "roll": 0.0},
        "size": {"x": 12.0, "y": 10.0, "z": 18.0},
        "snap_to_ground": True
    },
    {
        "id": "marmara_default_03",
        "type": "static.prop.streetbarrier",
        "location": {"x": 41.82372283935547, "y": -99.88819885253906, "z": 2.1514739990234375},
        "rotation": {"pitch": 0.0, "yaw": 20, "roll": 0.0},
        "size": {"x": 12.0, "y": 10.0, "z": 18.0},
        "snap_to_ground": True
    },
    {
        "id": "marmara_default_04",
        "type": "static.prop.streetbarrier",
        "location": {"x": 31.510379791259766, "y": -99.12989807128906, "z": 2.1519150733947754},
        "rotation": {"pitch": 0.0, "yaw": 190, "roll": 0.0},
        "size": {"x": 12.0, "y": 10.0, "z": 18.0},
        "snap_to_ground": True
    },
    {
        "id": "marmara_default_05",
        "type": "static.prop.streetbarrier",
        "location": {"x": 1.01917724609, "y": 133.81707763671875, "z": 2.2669081687927246},
        "rotation": {"pitch": 0.0, "yaw": 80, "roll": 0.0},
        "size": {"x": 12.0, "y": 10.0, "z": 18.0},
        "snap_to_ground": True
    },
    {
        "id": "marmara_default_06",
        "type": "static.prop.streetbarrier",
        "location": {"x": -2.822889566421509, "y": 143.71510314941406, "z": 2.1508734226226807},
        "rotation": {"pitch": 0.0, "yaw": 110, "roll": 0.0},
        "size": {"x": 12.0, "y": 10.0, "z": 18.0},
        "snap_to_ground": True
    },
    {
        "id": "marmara_default_07",
        "type": "static.prop.streetbarrier",
        "location": {"x": -119.1131591796875, "y": 240.69735717773438, "z": 2.174759864807129},
        "rotation": {"pitch": 0.0, "yaw": 80, "roll": 0.0},
        "size": {"x": 12.0, "y": 10.0, "z": 18.0},
        "snap_to_ground": True
    },
    {
        "id": "marmara_default_08",
        "type": "static.prop.streetbarrier",
        "location": {"x": -114.1131591796875, "y": 232.31048583984375, "z": 2.1512818336486816},
        "rotation": {"pitch": 0.0, "yaw": 150, "roll": 0.0},
        "size": {"x": 12.0, "y": 10.0, "z": 18.0},
        "snap_to_ground": True
    }
]

DEFAULT_BLUEPRINT_FILTERS = [
    "*building*",
    "*build*",
    "*house*",
    "*home*",
    "*apartment*",
    "*apart*",
    "*block*",
    "*mansion*",
    "*tower*",
    "*skyscraper*",
    "*skycraper*",
    "*sky*",
    "*office*",
    "*store*",
    "*villa*",
]

PROP_FALLBACK_FILTERS = [
    "static.*",
]


def load_building_specs(path):
    if not path:
        return DEFAULT_BUILDINGS

    with open(path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("buildings"), list):
        return data["buildings"]

    raise ValueError("Building JSON must be a list or an object with a 'buildings' list.")


def as_float(value, default=0.0):
    if value is None:
        return default
    return float(value)


def make_transform(spec):
    location = spec.get("location", spec)
    rotation = spec.get("rotation", {})
    return carla.Transform(
        carla.Location(
            x=as_float(location.get("x")),
            y=as_float(location.get("y")),
            z=as_float(location.get("z"))),
        carla.Rotation(
            pitch=as_float(rotation.get("pitch")),
            yaw=as_float(rotation.get("yaw")),
            roll=as_float(rotation.get("roll"))))


def set_scale(actor, spec):
    scale = spec.get("scale")
    if not scale:
        return

    actor.set_transform(carla.Transform(
        actor.get_transform().location,
        actor.get_transform().rotation))
    actor.set_simulate_physics(False)

    try:
        actor.set_scale(carla.Vector3D(
            x=as_float(scale.get("x"), 1.0),
            y=as_float(scale.get("y"), 1.0),
            z=as_float(scale.get("z"), 1.0)))
    except AttributeError:
        print("Warning: this CARLA build does not expose actor.set_scale(); scale ignored.")


def find_blueprints(world, filters):
    library = world.get_blueprint_library()
    matches = []
    seen = set()

    for bp_filter in filters:
        for blueprint in library.filter(bp_filter):
            if blueprint.id not in seen:
                seen.add(blueprint.id)
                matches.append(blueprint)

    return matches


def choose_blueprint(world, spec, allow_prop_fallback=False, override_filter=None):
    library = world.get_blueprint_library()
    building_type = override_filter or spec.get("type") or spec.get("blueprint") or "auto"

    if building_type != "auto":
        matches = library.filter(building_type)
        if not matches:
            raise RuntimeError("No blueprint matched '%s'." % building_type)
        return matches[0]

    candidates = find_blueprints(world, DEFAULT_BLUEPRINT_FILTERS)
    if candidates:
        return candidates[0]

    if allow_prop_fallback:
        fallback_candidates = find_blueprints(world, PROP_FALLBACK_FILTERS)
        if fallback_candidates:
            return fallback_candidates[0]

    return None


def snap_transform_to_ground(world, transform, above_ground):
    location = transform.location

    try:
        projected = world.ground_projection(location, 200.0)
        if projected is not None:
            transform.location.z = projected.location.z + above_ground
            return transform
    except AttributeError:
        pass

    try:
        start = carla.Location(location.x, location.y, location.z + 100.0)
        end = carla.Location(location.x, location.y, location.z - 200.0)
        hits = world.cast_ray(start, end)
        if hits:
            transform.location.z = hits[0].location.z + above_ground
    except AttributeError:
        pass

    return transform


def describe_blueprint(blueprint):
    attributes = []
    for attribute in blueprint:
        try:
            attributes.append("%s=%s" % (attribute.id, attribute.as_string()))
        except RuntimeError:
            attributes.append(attribute.id)

    if attributes:
        return "%s  [%s]" % (blueprint.id, ", ".join(attributes))
    return blueprint.id


def list_blueprints(world, filters=None, with_attributes=False):
    if filters:
        blueprints = find_blueprints(world, filters)
        if not blueprints:
            print("No blueprints matched: %s" % ", ".join(filters))
            return

        for blueprint in blueprints:
            print(describe_blueprint(blueprint) if with_attributes else blueprint.id)
        return

    blueprints = find_blueprints(world, DEFAULT_BLUEPRINT_FILTERS)
    fallback_blueprints = find_blueprints(world, PROP_FALLBACK_FILTERS)

    if blueprints:
        print("Building-like blueprints:")
        for blueprint in blueprints:
            text = describe_blueprint(blueprint) if with_attributes else blueprint.id
            print("  %s" % text)
    else:
        print("No building-like blueprints found.")

    if fallback_blueprints:
        print("Static prop fallback blueprints:")
        for blueprint in fallback_blueprints:
            text = describe_blueprint(blueprint) if with_attributes else blueprint.id
            print("  %s" % text)

    if not blueprints and not fallback_blueprints:
        print("No matching building/static prop blueprints found.")


def apply_blueprint_attributes(blueprint, attributes):
    for key, value in attributes.items():
        if blueprint.has_attribute(key):
            blueprint.set_attribute(key, str(value))
        else:
            print("Warning: blueprint '%s' has no attribute '%s'; ignored." %
                  (blueprint.id, key))


def draw_debug_building_box(world, label, transform, spec, life_time):
    size = spec.get("size", {"x": 12.0, "y": 10.0, "z": 18.0})
    height = as_float(size.get("z"), 18.0)
    center = carla.Location(
        x=transform.location.x,
        y=transform.location.y,
        z=transform.location.z + (height * 0.5))
    extent = carla.Vector3D(
        x=as_float(size.get("x"), 12.0) * 0.5,
        y=as_float(size.get("y"), 10.0) * 0.5,
        z=height * 0.5)
    box = carla.BoundingBox(center, extent)

    world.debug.draw_box(
        box,
        transform.rotation,
        thickness=0.20,
        color=carla.Color(255, 80, 0),
        life_time=life_time)
    world.debug.draw_string(
        carla.Location(center.x, center.y, center.z + extent.z + 1.0),
        label,
        draw_shadow=True,
        color=carla.Color(255, 255, 255),
        life_time=life_time)


def spawn_buildings(world, specs, dry_run=False, draw_debug_boxes=False,
                    debug_life_time=600.0, allow_prop_fallback=False,
                    blueprint_filter=None):
    spawned = []
    missing_blueprint_warning_printed = False

    for index, spec in enumerate(specs, start=1):
        blueprint = choose_blueprint(
            world,
            spec,
            allow_prop_fallback=allow_prop_fallback,
            override_filter=blueprint_filter)
        transform = make_transform(spec)

        if spec.get("snap_to_ground", True):
            transform = snap_transform_to_ground(
                world,
                transform,
                as_float(spec.get("above_ground"), 0.05))

        label = spec.get("id", "building_%03d" % index)
        scale = spec.get("scale", {"x": 1.0, "y": 1.0, "z": 1.0})
        print(
            "%s %s at x=%.2f y=%.2f z=%.2f yaw=%.2f scale=(%.2f, %.2f, %.2f)" %
            (
                "Would spawn" if dry_run else "Spawning",
                label,
                transform.location.x,
                transform.location.y,
                transform.location.z,
                transform.rotation.yaw,
                as_float(scale.get("x"), 1.0),
                as_float(scale.get("y"), 1.0),
                as_float(scale.get("z"), 1.0),
            ))

        if draw_debug_boxes and not dry_run:
            draw_debug_building_box(world, label, transform, spec, debug_life_time)
            print("  drew debug building box for location checking")

        if blueprint is None:
            if not missing_blueprint_warning_printed:
                print(
                    "  No building-like runtime blueprint found. "
                    "Only debug boxes were drawn. Use --list-building-blueprints "
                    "to inspect available blueprints, or use the Unreal editor "
                    "script to save real mesh buildings into the map.")
                missing_blueprint_warning_printed = True
            continue

        apply_blueprint_attributes(blueprint, spec.get("attributes", {}))

        if dry_run:
            continue

        actor = world.spawn_actor(blueprint, transform)
        actor.set_simulate_physics(False)
        set_scale(actor, spec)
        spawned.append(actor)
        print("  actor_id=%d blueprint=%s" % (actor.id, blueprint.id))

    return spawned


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    argparser.add_argument("-p", "--port", default=2000, type=int, help="CARLA port")
    argparser.add_argument("--timeout", default=10.0, type=float, help="Client timeout in seconds")
    argparser.add_argument(
        "--map",
        default=None,
        help="Optional map name to load first, e.g. MarmaraIntersection_v011")
    argparser.add_argument(
        "--config",
        default=None,
        help="JSON file with building placements. Uses the built-in Marmara default if omitted.")
    argparser.add_argument(
        "--list-building-blueprints",
        action="store_true",
        help="Print candidate building/static-prop blueprint ids and exit.")
    argparser.add_argument(
        "--list-blueprints",
        nargs="*",
        metavar="FILTER",
        help="Print blueprint ids matching one or more filters, e.g. '*building*' 'static.prop.*'.")
    argparser.add_argument(
        "--with-attributes",
        action="store_true",
        help="Include blueprint attributes when listing blueprints.")
    argparser.add_argument(
        "--blueprint-filter",
        default=None,
        help="Force this blueprint filter for every building placement, e.g. 'static.prop.building*'.")
    argparser.add_argument("--dry-run", action="store_true", help="Validate and print placements only.")
    argparser.add_argument(
        "--no-debug-boxes",
        action="store_true",
        help="Do not draw orange debug boxes at the requested building footprints.")
    argparser.add_argument(
        "--debug-life-time",
        default=600.0,
        type=float,
        help="How long debug building boxes stay visible, in seconds.")
    argparser.add_argument(
        "--allow-prop-fallback",
        action="store_true",
        help="Allow generic static.prop.* actors if no building-like blueprint exists.")
    argparser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep the script alive until Ctrl+C, then destroy spawned actors.")
    args = argparser.parse_args()

    load_carla_module()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = client.load_world(args.map) if args.map else client.get_world()
    print("Connected to map: %s" % world.get_map().name)

    if args.list_building_blueprints:
        list_blueprints(world, with_attributes=args.with_attributes)
        return

    if args.list_blueprints is not None:
        filters = args.list_blueprints if args.list_blueprints else ["*"]
        list_blueprints(world, filters=filters, with_attributes=args.with_attributes)
        return

    specs = load_building_specs(args.config)
    spawned = spawn_buildings(
        world,
        specs,
        dry_run=args.dry_run,
        draw_debug_boxes=not args.no_debug_boxes,
        debug_life_time=args.debug_life_time,
        allow_prop_fallback=args.allow_prop_fallback,
        blueprint_filter=args.blueprint_filter)

    if args.keep_alive and spawned:
        print("Spawned %d building actor(s). Press Ctrl+C to destroy them." % len(spawned))
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            for actor in spawned:
                actor.destroy()
            print("Destroyed spawned building actor(s).")


if __name__ == "__main__":
    main()
