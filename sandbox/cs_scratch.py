import moderngl as mgl
from PIL import Image

ctx = mgl.create_standalone_context()

compute_shader = ctx.compute_shader(
    '''
        #version 430
        layout (local_size_x = 16, local_size_y = 16) in;
        // match the input texture format!
        layout(rgba8, location=0) writeonly uniform image2D destTex;
        
        void main() {
            // texel coordinate we are writing to
            ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
            // Calculate 1.0 - distance from the center in each work group
            float local = 1.0 - length(vec2(ivec2(gl_LocalInvocationID.xy) - 8) / 8.0);
            // Wave covering the screen diagonally
            float global = sin(float(gl_WorkGroupID.x + gl_WorkGroupID.y) * 0.1 + 0.1) / 2.0 + 0.5;
            imageStore(destTex, texelPos, vec4(local, global, 0.0, 1.0));
        }
    '''
)
compute_shader['destTex'] = 0

compute_shader_2 = ctx.compute_shader(
    '''
        #version 430
        layout (local_size_x = 16, local_size_y = 16) in;
        
        layout(rgba8, location=1) readonly uniform image2D origTex;
        layout(rgba8, location=2) writeonly uniform image2D destTex;
        
        void main() {
            ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
            
            imageStore(destTex, texelPos, imageLoad(origTex, texelPos) * vec4(vec3(0.5), 1.0));
        }
        
    '''
)
compute_shader_2['origTex'] = 0
compute_shader_2['destTex'] = 1

tex = ctx.texture((1280, 720), 4)
tex.filter = mgl.NEAREST, mgl.NEAREST

w, h = tex.size
gw, gh = 16, 16
nx, ny, nz = int(w / gw), int(h / gh), 1

tex.bind_to_image(0, read=False, write=True)
compute_shader.run(nx, ny, nz)

image = Image.frombytes('RGBA', tex.size, tex.read(), 'raw', 'RGBA', 0, -1)
image.show()

tex2 = ctx.texture((1280, 720), 4)
tex2.filter = mgl.NEAREST, mgl.NEAREST

w, h = tex2.size
gw, gh = 16, 16
nx, ny, nz = int(w / gw), int(h / gh), 1

tex.bind_to_image(0, read=True, write=False)
tex2.bind_to_image(1, read=False, write=True)
compute_shader_2.run(nx, ny, nz)

image = Image.frombytes('RGBA', tex2.size, tex2.read(), 'raw', 'RGBA', 0, -1)
image.show()

tex.release()
tex2.release()

compute_shader.release()
compute_shader_2.release()
