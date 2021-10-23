from ..utils.Constants import *
from ..utils.utils import *
# import glm

class Camera:
    def __init__(self, core_component):
        self.core_component = core_component
        self.look_at = np.array([0, 0, 0], dtype=np.float32)
        self.eye = np.array([1.2, 1.0, 2.0], dtype=np.float32)
        self.eye_up = np.array([0, 1, 0], dtype=np.float32)

        # Perspective params for camera
        self.width, self.height = global_screen_width, global_screen_height
        self.aspect = self.width / self.height
        self.near = 0.1
        self.far = 100.0
        self.fov = 45

        # Ortho params for shadow map
        self.ortho_left, self.ortho_right, self.ortho_bottom, self.ortho_top = -10.0, 10.0, -10.0, 10.0
        self.ortho_near, self.ortho_far = 1.0, 20.0
        self.ortho_p_matrix = np.eye(4, dtype=default_dtype)

        self.p_matrix = np.eye(4, dtype=default_dtype)
        self.v_matrix = np.eye(4, dtype=default_dtype)
        self.m_matrix = np.eye(4, dtype=default_dtype)
        self.mvp_matrix = np.eye(4, dtype=default_dtype)

        self.need_update = True
        self.initialized = False

    def initialization(self):
        self.need_update = True
        self.p_matrix = self.get_projection_mat(self.need_update)
        self.v_matrix = self.get_view_mat(self.need_update)
        self.m_matrix = self.get_model_mat(self.need_update)
        self.mvp_matrix = self.get_modelviewprojection_mat(need_update=self.need_update)

        self.ortho_p_matrix = self.get_ortho_projection_mat(self.need_update)

        self.initialized = True

    def get_look_at_params(self):
        return self.eye, self.look_at, self.eye_up

    def get_perspective_params(self):
        return self.fov, self.aspect, self.near, self.far

    def get_ortho_params(self):
        return self.ortho_left, self.ortho_right, self.ortho_bottom, self.ortho_top, self.ortho_near, self.ortho_far

    def get_model_mat(self, need_update=False):
        m_mat = np.eye(4, dtype=np.float32) if need_update else self.m_matrix
        return m_mat

    def get_projection_mat(self, need_update=False):
        p_mat = compute_perspective_mat(self.fov, self.aspect, self.near, self.far, np.float32) if need_update else self.p_matrix
        return p_mat

    def get_ortho_projection_mat(self, need_update=False):
        ortho_mat = compute_ortho_mat(self.ortho_left, self.ortho_right, self.ortho_bottom, self.ortho_top,
                                      self.ortho_near, self.ortho_far) if need_update else self.ortho_p_matrix
        return ortho_mat

    def get_view_mat(self, need_update=False):
        v_mat = compute_lookat_mat(self.eye, self.look_at, self.eye_up, np.float32) if need_update else self.v_matrix
        return v_mat

    def get_modelviewprojection_mat(self, need_update=False):
        mvp = self.get_model_mat(need_update).dot(self.get_view_mat(need_update)).dot(self.get_projection_mat(need_update))
        return mvp

    def get_modelview_mat(self, need_update=False):
        mv = self.get_model_mat(need_update).dot(self.get_view_mat(need_update))
        return mv

    def set_eye_position(self, new_eye):
        self.update_mvp(eye=new_eye)

    def set_fov(self, new_fov):
        self.update_mvp(fov=new_fov)

    def update_mvp(self, fov=None, aspect=None, near=None, far=None, eye=None):
        self.need_update = False
        if fov is not None and fov > 0 and fov != self.fov:
            self.fov = fov
            self.need_update = True
        if aspect is not None and aspect > 0 and aspect != self.aspect:
            self.aspect = aspect
            self.need_update = True
        if near is not None and near > 0 and near != self.near:
            self.near = near
            self.need_update = True
        if far is not None and far > 0 and far != self.far:
            self.far = far
            self.need_update = True
        if eye is not None and (self.eye - eye).any():
            self.eye = eye
            self.need_update = True

        if self.need_update:
            self.p_matrix = self.get_projection_mat(self.need_update)
            self.v_matrix = self.get_view_mat(self.need_update)
            self.m_matrix = self.get_model_mat(self.need_update)
            self.mvp_matrix = self.get_modelviewprojection_mat(need_update=False)
            self.need_update = False


    def update(self):
        pass