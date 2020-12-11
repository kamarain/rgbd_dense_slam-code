//uniform vec4 diffuseMat;
//uniform vec4 specMat;
//uniform float specPow;

uniform vec4 globalLight;

varying vec3 N;
varying vec3 c;
varying vec3 L;
varying vec3 E;
varying float attenuation;

attribute vec3 inputVertex;
attribute vec3 inputColor;
attribute vec3 inputNormal;


void main(void)
{
   vec3 v = vec3(gl_ModelViewMatrix * vec4(inputVertex,1.0));
   N = -normalize(gl_NormalMatrix * inputNormal);
   c = inputColor;
   vec3 lightDir = globalLight.xyz/*gl_LightSource[0].position.xyz*/ - v; 
   L = normalize(lightDir);
   E = -normalize(v);
   gl_Position = gl_ModelViewProjectionMatrix *  vec4(inputVertex,1.0);
   attenuation = 1.0;//clamp(dot(lightDir,lightDir)),0.0,1.0);
   
}

