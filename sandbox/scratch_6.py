"""
Compute shader renders a 32 x 32 grid to a 512, 512 texture
"""
import sys

import moderngl as mgl
import moderngl_window as mglw
from PIL import Image
from moderngl_window import geometry


class Example(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "ModernGL Example"
    window_size = (1280, 720)
    aspect_ratio = 16 / 9
    resizable = True


class RenderTextureCompute(Example):
    title = "Render Texture Using Compute Shader"
    gl_version = (4, 3)
    aspect_ratio = 1.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compute = self.ctx.compute_shader('''
            #version 430
            layout (local_size_x = 32, local_size_y = 32) in;
            // match the input texture format!
            layout(rgba8, location=0) writeonly uniform image2D destTex;
            uniform float time;
            void main() {
                // texel coordinate we are writing to
                ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
                // Calculate 1.0 - distance from the center in each work group
                float local = 1.0 - length(vec2(ivec2(gl_LocalInvocationID.xy) - 16) / 16.0);
                // Wave covering the screen diagonally
                float global = sin(float(gl_WorkGroupID.x + gl_WorkGroupID.y) * 0.1 + time) / 2.0 + 0.5;
                imageStore(
                    destTex,
                    texelPos,
                    vec4(
                        local,
                        global,
                        0.0,
                        1.0
                    )
                );
            }
        ''')
        self.compute['destTex'] = 0

        # For rendering a simple textured quad
        self.quad_program = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec3 in_position;
            in vec2 in_texcoord_0;
            out vec2 uv;
            void main() {
                gl_Position = vec4(in_position, 1.0);
                uv = in_texcoord_0;
            }
            """,
            fragment_shader="""
            #version 330
            uniform sampler2D texture0;
            out vec4 fragColor;
            in vec2 uv;
            void main() {
                fragColor = texture(texture0, uv);
            }
            """,
        )

        # RGB_8 texture
        self.texture = self.ctx.texture((1280, 720), 4)
        self.texture.filter = mgl.NEAREST, mgl.NEAREST
        self.quad_fs = geometry.quad_fs()

    def render(self, time, frame_time):
        self.ctx.clear(0.3, 0.3, 0.3)

        w, h = self.texture.size
        gw, gh = 16, 16
        nx, ny, nz = int(w / gw), int(h / gh), 1

        try:
            self.compute['time'] = time
        except Exception:
            pass

        # Automatically binds as a GL_R32F / r32f (read from the texture)
        self.texture.bind_to_image(0, read=False, write=True)
        self.compute.run(nx, ny, nz)

        image = Image.frombytes('RGBA', self.texture.size, self.texture.read(), 'raw', 'RGBA', 0, -1)
        image.save("output.png")

        # Render texture
        self.texture.use(location=0)
        self.quad_fs.render(self.quad_program)

        sys.exit(-1)


if __name__ == '__main__':
    RenderTextureCompute.run()
