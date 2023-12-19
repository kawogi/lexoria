use glam::Vec3;
use rand::Rng;

use crate::{Mesh, Vertex};

fn vertex(pos: [i16; 3], nor: [i8; 3], color: Vec3) -> Vertex {
    Vertex {
        _pos: [pos[0], pos[1], pos[2], 1],
        _normal: [nor[0], nor[1], nor[2], 0],
        _color: [color[0], color[1], color[2], 1.0],
    }
}

#[allow(clippy::too_many_lines)]
pub(crate) fn create_world() -> Mesh {
    const X: usize = 32;
    const Y: usize = 32;
    const Z: usize = 8;
    const HX: i16 = X as i16 / 2;
    const HY: i16 = Y as i16 / 2;
    const HZ: i16 = Z as i16 / 2;

    let mut rng = rand::thread_rng();

    let mut height_map = vec![vec![0.0; X]; Y];
    let mut max = 0.0_f32;
    for (y, heights) in height_map.iter_mut().enumerate() {
        for (x, height) in heights.iter_mut().enumerate() {
            let xx = x as f32 / X as f32;
            let yy = y as f32 / Y as f32;
            let h = (xx * 2.0 + 1.6).sin() * (yy * 3.0 + 0.8).sin();
            let h = h + (xx * 7.0 + 0.7).sin() * (yy * 6.6 + 1.3).sin() * 0.5;
            let h = h.max(0.0) + rng.gen::<f32>() * 0.1;
            max = max.max(h);
            *height = h;
        }
    }
    for heights in &mut height_map {
        for height in heights {
            *height *= Z as f32 / max;
        }
    }

    let colors = [
        Vec3::new(0.2, 0.6, 0.8),
        Vec3::new(1.0, 1.0, 0.5),
        Vec3::new(0.6, 0.8, 0.0),
        Vec3::new(0.2, 0.7, 0.1),
        Vec3::new(0.3, 0.7, 0.1),
        Vec3::new(0.7, 0.3, 0.2),
        Vec3::new(0.7, 0.7, 0.7),
        Vec3::new(1.0, 1.0, 1.0),
    ];

    let mut voxels = [[[Option::<Vec3>::None; X]; Y]; Z];
    for (z, xy_plane) in voxels.iter_mut().enumerate() {
        for (y, x_column) in xy_plane.iter_mut().enumerate() {
            for (x, voxel) in x_column.iter_mut().enumerate() {
                *voxel = ((z as f32) < height_map[y][x]).then_some(colors[z]);
            }
        }
    }

    let get_voxel = |x: i16, y: i16, z: i16| -> _ {
        let Ok(x) = usize::try_from(x + HX) else {
            return None;
        };
        let Ok(y) = usize::try_from(y + HY) else {
            return None;
        };
        let Ok(z) = usize::try_from(z + HZ) else {
            return None;
        };
        let Some(xy_plane) = voxels.get(z) else {
            return None;
        };
        let Some(x_column) = xy_plane.get(y) else {
            return None;
        };
        let Some(&voxel) = x_column.get(x) else {
            return None;
        };

        voxel
    };

    let mut vertices = Vec::new();

    for x in -HX..=HX {
        for z in -HZ..HZ {
            for y in -HY..HY {
                let is_a = get_voxel(x - 1, y, z);
                let is_b = get_voxel(x, y, z);
                match (is_a, is_b) {
                    (None, Some(color)) => {
                        let normal = [-1, 0, 0];
                        vertices.push(vertex([x, y, z], normal, color));
                        vertices.push(vertex([x, y + 1, z], normal, color));
                        vertices.push(vertex([x, y, z + 1], normal, color));
                        vertices.push(vertex([x, y + 1, z + 1], normal, color));
                    }
                    (Some(color), None) => {
                        let normal = [1, 0, 0];
                        vertices.push(vertex([x, y, z], normal, color));
                        vertices.push(vertex([x, y, z + 1], normal, color));
                        vertices.push(vertex([x, y + 1, z], normal, color));
                        vertices.push(vertex([x, y + 1, z + 1], normal, color));
                    }
                    _ => {}
                }
            }
        }
    }

    for y in -HY..=HY {
        for x in -HX..HX {
            for z in -HZ..HZ {
                let is_a = get_voxel(x, y - 1, z);
                let is_b = get_voxel(x, y, z);
                match (is_a, is_b) {
                    (None, Some(color)) => {
                        let normal = [0, -1, 0];
                        vertices.push(vertex([x, y, z], normal, color));
                        vertices.push(vertex([x, y, z + 1], normal, color));
                        vertices.push(vertex([x + 1, y, z], normal, color));
                        vertices.push(vertex([x + 1, y, z + 1], normal, color));
                    }
                    (Some(color), None) => {
                        let normal = [0, 1, 0];
                        vertices.push(vertex([x, y, z], normal, color));
                        vertices.push(vertex([x + 1, y, z], normal, color));
                        vertices.push(vertex([x, y, z + 1], normal, color));
                        vertices.push(vertex([x + 1, y, z + 1], normal, color));
                    }
                    _ => {}
                }
            }
        }
    }

    for z in -HZ..=HZ {
        for y in -HY..HY {
            for x in -HX..HX {
                let is_a = get_voxel(x, y, z - 1);
                let is_b = get_voxel(x, y, z);
                match (is_a, is_b) {
                    (None, Some(color)) => {
                        let normal = [0, 0, -1];
                        vertices.push(vertex([x, y, z], normal, color));
                        vertices.push(vertex([x + 1, y, z], normal, color));
                        vertices.push(vertex([x, y + 1, z], normal, color));
                        vertices.push(vertex([x + 1, y + 1, z], normal, color));
                    }
                    (Some(color), None) => {
                        let normal = [0, 0, 1];
                        vertices.push(vertex([x, y, z], normal, color));
                        vertices.push(vertex([x, y + 1, z], normal, color));
                        vertices.push(vertex([x + 1, y, z], normal, color));
                        vertices.push(vertex([x + 1, y + 1, z], normal, color));
                    }
                    _ => {}
                }
            }
        }
    }

    // let quad_count = (vertex_data.len() / 4) as u16;
    let indices = (0..vertices.len())
        .map(|index| index as u16 & !0b11)
        .flat_map(|index| [index, index + 2, index + 1, index + 1, index + 2, index + 3])
        .collect::<Vec<_>>();

    Mesh { vertices, indices }
}
