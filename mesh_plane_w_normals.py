
    triangle_normals = np.asarray(room_mesh.triangle_normals)
    triangle_normals = triangle_normals / np.linalg.norm(triangle_normals, axis=1)[:, np.newaxis]

    threshold = 0.1
    num_triangles = len(triangle_normals)
    labels = -np.ones(len(triangle_normals), dtype=int)
    cluster_id = 0

    for i in range(num_triangles):
        if labels[i] != -1:
            continue  # This triangle is already clustered
        labels[i] = cluster_id
        for j in range(i + 1, num_triangles):
            if cosine(triangle_normals[i], triangle_normals[j]) < threshold:
                labels[j] = cluster_id
        cluster_id += 1

    # Number of clusters
    num_clusters = cluster_id
    print(f"Number of clusters: {num_clusters}")

    # Assign cluster labels to triangles
    triangle_clusters = [[] for _ in range(num_clusters)]
    for i, label in enumerate(labels):
        triangle_clusters[label].append(i)

    # Visualize each cluster
    for i, cluster in enumerate(triangle_clusters):
        cluster_mesh = room_mesh.select_by_index(cluster, triangle=True)
        cluster_mesh.paint_uniform_color(np.random.rand(3))  # Random color for visualization
        o3d.visualization.draw_geometries([cluster_mesh], window_name=f"Cluster {i+1}", width=800, height=600)