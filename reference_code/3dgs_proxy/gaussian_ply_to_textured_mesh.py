#!/usr/bin/env python3
"""
Convert a Gaussian-splat PLY into a continuous textured mesh for UE5.

Pipeline:
1. Auto-crop the largest high-opacity object cluster.
2. Reconstruct a continuous surface from the cropped Gaussian centers.
3. Transfer Gaussian DC color to mesh vertices.
4. Use Blender to unwrap UVs, bake vertex colors to a PNG texture, and export FBX.

This produces a conventional Static Mesh asset. It is not an exact Gaussian
renderer, but it is the closest fit for "a formed mesh body with a texture".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from gaussian_ply_to_splat_mesh import apply_bounds, largest_cluster_bounds, parse_vec3
from ply_to_mesh import SH_C0, read_vertex_ply, sigmoid


def load_cropped_points(path: Path, args: argparse.Namespace):
    vertices, names = read_vertex_ply(path)
    required = {"x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"}
    missing = sorted(required.difference(names))
    if missing:
        raise ValueError(f"Input PLY is missing required fields: {missing}")

    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)
    opacity = sigmoid(vertices["opacity"].astype(np.float32))
    colors = np.clip(
        0.5 + SH_C0 * np.column_stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]).astype(np.float32),
        0.0,
        1.0,
    )

    crop_min = args.crop_min
    crop_max = args.crop_max
    auto_bounds = None
    if args.auto_crop_largest_cluster:
        auto_min, auto_max = largest_cluster_bounds(points, opacity, args)
        crop_min = tuple(auto_min.tolist()) if crop_min is None else tuple(np.maximum(auto_min, np.array(crop_min)).tolist())
        crop_max = tuple(auto_max.tolist()) if crop_max is None else tuple(np.minimum(auto_max, np.array(crop_max)).tolist())
        auto_bounds = (np.array(crop_min, dtype=np.float32), np.array(crop_max, dtype=np.float32))

    keep = np.isfinite(points).all(axis=1)
    keep &= opacity >= float(args.min_opacity)
    keep = apply_bounds(points, keep, crop_min, crop_max)
    idx = np.where(keep)[0]
    if args.max_points and len(idx) > args.max_points:
        order = np.argsort(opacity[idx])[::-1]
        idx = idx[order[: args.max_points]]

    return points[idx], colors[idx], opacity[idx], auto_bounds


def make_point_cloud(points: np.ndarray, colors: np.ndarray, args: argparse.Namespace) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    if args.voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(float(args.voxel_size))

    if args.remove_outliers:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.0)

    dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    dists = dists[np.isfinite(dists) & (dists > 0)]
    median_dist = float(np.median(dists)) if len(dists) else 0.01
    normal_radius = max(median_dist * 8.0, 0.015)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=50))
    try:
        pcd.orient_normals_consistent_tangent_plane(int(args.orientation_neighbors))
    except RuntimeError as exc:
        print(f"[warn] normal orientation failed: {exc}")
    return pcd


def reconstruct_mesh(pcd: o3d.geometry.PointCloud, args: argparse.Namespace) -> o3d.geometry.TriangleMesh:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=int(args.depth),
        scale=float(args.poisson_scale),
        linear_fit=bool(args.linear_fit),
    )

    densities = np.asarray(densities)
    if 0.0 < args.density_trim_quantile < 1.0:
        keep = densities > np.quantile(densities, float(args.density_trim_quantile))
        mesh = mesh.select_by_index(np.where(keep)[0])

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    if args.keep_largest:
        clusters, counts, _ = mesh.cluster_connected_triangles()
        clusters = np.asarray(clusters)
        counts = np.asarray(counts)
        if len(counts):
            largest = int(np.argmax(counts))
            mesh.remove_triangles_by_mask(clusters != largest)
            mesh.remove_unreferenced_vertices()

    if args.smooth_iterations > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=int(args.smooth_iterations))

    if args.simplify_target_faces and len(mesh.triangles) > args.simplify_target_faces:
        mesh = mesh.simplify_quadric_decimation(int(args.simplify_target_faces))

    mesh.compute_vertex_normals()
    return mesh


def transfer_colors(mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud, args: argparse.Namespace) -> np.ndarray:
    source_points = np.asarray(pcd.points)
    source_colors = np.asarray(pcd.colors)
    target_points = np.asarray(mesh.vertices)

    tree = o3d.geometry.KDTreeFlann(pcd)
    out = np.empty((len(target_points), 3), dtype=np.float32)
    k = max(1, int(args.color_neighbors))
    for i, point in enumerate(target_points):
        _, idx, dist2 = tree.search_knn_vector_3d(point, k)
        idx = np.asarray(idx, dtype=np.int64)
        dist2 = np.asarray(dist2, dtype=np.float64)
        weights = 1.0 / np.maximum(dist2, 1e-8)
        weights /= weights.sum()
        out[i] = (source_colors[idx] * weights[:, None]).sum(axis=0)
    return np.clip(out, 0.0, 1.0)


def o3d_to_arrays(mesh: o3d.geometry.TriangleMesh, vertex_colors: np.ndarray):
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    return vertices, faces, vertex_colors.astype(np.float32)


def export_textured_fbx_with_blender(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    fbx_path: Path,
    texture_path: Path,
    name: str,
    texture_size: int,
) -> None:
    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update(calc_edges=False)

    color_attr = mesh.color_attributes.new(name="Color", type="BYTE_COLOR", domain="CORNER")
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            r, g, b = colors[vertex_index]
            color_attr.data[loop_index].color = (float(r), float(g), float(b), 1.0)

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mat = bpy.data.materials.new(f"M_{name}_Baked")
    mat.use_nodes = True
    obj.data.materials.append(mat)

    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    attr_node = nodes.new(type="ShaderNodeAttribute")
    attr_node.attribute_name = "Color"
    tex_node = nodes.new(type="ShaderNodeTexImage")

    image = bpy.data.images.new(f"T_{name}_BaseColor", width=int(texture_size), height=int(texture_size))
    image.generated_color = (0.5, 0.5, 0.5, 1.0)
    image.filepath_raw = str(texture_path.resolve())
    image.file_format = "PNG"
    tex_node.image = image
    nodes.active = tex_node

    if bsdf is not None:
        mat.node_tree.links.new(attr_node.outputs["Color"], bsdf.inputs["Base Color"])
        try:
            bsdf.inputs["Roughness"].default_value = 0.85
        except Exception:
            pass

    bpy.ops.object.shade_smooth()
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15192, island_margin=0.01)
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 16
    bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.view_settings.look = "None"
    bpy.context.scene.view_settings.exposure = 0.0
    bpy.context.scene.view_settings.gamma = 1.0

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, margin=8)
    except TypeError:
        bpy.ops.object.bake(type="DIFFUSE", margin=8)

    texture_path.parent.mkdir(parents=True, exist_ok=True)
    image.save()

    # Switch the exported material to the baked texture.
    if bsdf is not None:
        for link in list(mat.node_tree.links):
            if link.to_node == bsdf and link.to_socket == bsdf.inputs["Base Color"]:
                mat.node_tree.links.remove(link)
        mat.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

    fbx_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=str(fbx_path.resolve()),
        use_selection=True,
        object_types={"MESH"},
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_UNITS",
        axis_forward="-Z",
        axis_up="Y",
        use_mesh_modifiers=True,
        mesh_smooth_type="FACE",
        use_triangles=True,
        bake_anim=False,
        add_leaf_bones=False,
        path_mode="COPY",
        embed_textures=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a continuous textured UE5 FBX from Gaussian PLY.")
    parser.add_argument("input", nargs="?", default="assets/point_cloud.ply")
    parser.add_argument("--fbx", default="assets/ue5_point_cloud_object/PointCloudObject_TexturedMesh_UE5.fbx")
    parser.add_argument("--texture", default="assets/ue5_point_cloud_object/T_PointCloudObject_TexturedMesh_BaseColor.png")
    parser.add_argument("--name", default="PointCloudObject_TexturedMesh")
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--min-opacity", type=float, default=0.18)
    parser.add_argument("--max-points", type=int, default=25000)
    parser.add_argument("--voxel-size", type=float, default=0.0)
    parser.add_argument("--remove-outliers", action="store_true")
    parser.add_argument("--orientation-neighbors", type=int, default=30)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--poisson-scale", type=float, default=1.08)
    parser.add_argument("--linear-fit", action="store_true")
    parser.add_argument("--density-trim-quantile", type=float, default=0.12)
    parser.add_argument("--keep-largest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smooth-iterations", type=int, default=2)
    parser.add_argument("--simplify-target-faces", type=int, default=35000)
    parser.add_argument("--color-neighbors", type=int, default=8)
    parser.add_argument("--crop-min", type=parse_vec3, default=None)
    parser.add_argument("--crop-max", type=parse_vec3, default=None)
    parser.add_argument("--auto-crop-largest-cluster", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cluster-opacity", type=float, default=0.7)
    parser.add_argument("--cluster-eps", type=float, default=0.5)
    parser.add_argument("--cluster-min-points", type=int, default=10)
    parser.add_argument("--cluster-margin", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points, colors, _, auto_bounds = load_cropped_points(Path(args.input), args)
    pcd = make_point_cloud(points, colors, args)
    mesh = reconstruct_mesh(pcd, args)
    mesh_colors = transfer_colors(mesh, pcd, args)
    vertices, faces, vertex_colors = o3d_to_arrays(mesh, mesh_colors)
    export_textured_fbx_with_blender(
        vertices,
        faces,
        vertex_colors,
        Path(args.fbx),
        Path(args.texture),
        args.name,
        args.texture_size,
    )

    if auto_bounds is not None:
        print(f"[auto-crop] min={auto_bounds[0]} max={auto_bounds[1]}")
    print(f"[points] {len(points)}")
    print(f"[mesh] vertices={len(vertices)} faces={len(faces)}")
    print(f"[save] FBX {Path(args.fbx).resolve()}")
    print(f"[save] texture {Path(args.texture).resolve()}")


if __name__ == "__main__":
    main()
