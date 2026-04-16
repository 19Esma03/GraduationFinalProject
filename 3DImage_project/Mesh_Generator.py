"""
Create_Mesh_Fixed.py
Nokta bulutundan yüksek kaliteli mesh oluşturma.
İki yöntem:
  1. Poisson Surface Reconstruction (kapalı, pürüzsüz)
  2. Ball Pivoting Algorithm — BPA (açık yüzeyler için)
"""

import open3d as o3d
import numpy as np
import os

INPUT_PLY  = "Output_Models/fixed_cloud.ply"
OUTPUT_DIR = "Output_Models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_prepare(ply_path, voxel_size=0.01):
    """
    PLY dosyasını yükle, voxelize et, normalleri hesapla.
    """
    print(f"Yükleniyor: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"  Ham nokta sayısı: {len(pcd.points)}")

    # Voxel downsample — mesh için uniform yoğunluk şart
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"  Voxel sonrası: {len(pcd.points)}")

    # Outlier temizle
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    pcd = pcd.select_by_index(ind)
    print(f"  Temizleme sonrası: {len(pcd.points)}")

    # Normaller (Poisson için zorunlu)
    print("  Normaller hesaplanıyor...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(100)

    return pcd


def poisson_mesh(pcd, depth=9, density_threshold=0.01):
    """
    Poisson Surface Reconstruction.
    depth=9 → yüksek detay (daha fazla RAM gerektirir)
    density_threshold → düşük yoğunluklu yüzleri kaldır (0.01-0.05 arası dene)
    """
    print("\nPoisson mesh oluşturuluyor...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=0, scale=1.1, linear_fit=False
    )

    # Düşük yoğunluklu (uydurulmuş) üçgenleri kaldır
    densities = np.asarray(densities)
    threshold = np.quantile(densities, density_threshold)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Temizle
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    print(f"  Üçgen sayısı: {len(mesh.triangles)}")
    print(f"  Köşe sayısı : {len(mesh.vertices)}")
    return mesh


def bpa_mesh(pcd, radii=None):
    """
    Ball Pivoting Algorithm — açık yüzeyler ve ince detaylar için daha iyi.
    radii: büyük r = pürüzsüz ama eksik detay; küçük r = detaylı ama gürültülü
    """
    print("\nBPA mesh oluşturuluyor...")
    if radii is None:
        # Nokta bulutu yoğunluğuna göre otomatik radii
        distances = pcd.compute_nearest_neighbor_distance()
        avg_d = np.mean(distances)
        radii = [avg_d, avg_d * 2, avg_d * 4]
        print(f"  Otomatik radii: {[f'{r:.4f}' for r in radii]}")

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()

    print(f"  Üçgen sayısı: {len(mesh.triangles)}")
    return mesh


def save_and_show(mesh, name, output_dir):
    path_obj = os.path.join(output_dir, f"{name}.obj")
    path_ply = os.path.join(output_dir, f"{name}.ply")
    o3d.io.write_triangle_mesh(path_obj, mesh)
    o3d.io.write_triangle_mesh(path_ply, mesh)
    print(f"  Kaydedildi: {path_obj}")
    print(f"  Kaydedildi: {path_ply}")

    o3d.visualization.draw_geometries(
        [mesh],
        window_name=name,
        width=1280, height=720,
        mesh_show_back_face=True
    )


if __name__ == "__main__":
    pcd = load_and_prepare(INPUT_PLY, voxel_size=0.01)

    # Yöntem 1: Poisson (kapalı objeler için önerilen)
    mesh_poisson = poisson_mesh(pcd, depth=9, density_threshold=0.02)
    save_and_show(mesh_poisson, "poisson_mesh", OUTPUT_DIR)

    # Yöntem 2: BPA (ek seçenek — yorum satırını kaldır)
    # mesh_bpa = bpa_mesh(pcd)
    # save_and_show(mesh_bpa, "bpa_mesh", OUTPUT_DIR)

    print("\nMesh oluşturma tamamlandı.")
