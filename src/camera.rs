use std::f32::consts;

use glam::{Mat4, Vec3};

pub(crate) struct Camera {
    pub(crate) eye: Vec3,
    pub(crate) center: Vec3,
    pub(crate) aspect_ratio: f32,
    pub(crate) z_near: f32,
    pub(crate) z_far: f32,
    pub(crate) fov_y_radians: f32,
}

impl Camera {
    pub(crate) fn new(eye: Vec3, center: Vec3, aspect_ratio: f32) -> Self {
        Self {
            eye,
            center,
            aspect_ratio,
            z_near: 1.0,
            z_far: 50.0,
            fov_y_radians: consts::FRAC_PI_4,
        }
    }

    fn view_matrix(&self) -> Mat4 {
        glam::Mat4::look_at_rh(self.eye, self.center, glam::Vec3::Z)
    }

    fn projection_matrix(&self) -> Mat4 {
        glam::Mat4::perspective_rh(
            self.fov_y_radians,
            self.aspect_ratio,
            self.z_near,
            self.z_far,
        )
    }

    pub(crate) fn matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }
}
