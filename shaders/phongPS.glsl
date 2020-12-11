//uniform vec4 ambientMat;
//uniform vec4 diffuseMat;
//uniform vec4 specMat;
//uniform float specPow;
uniform vec4 globalLight;

varying vec3 N;
varying vec3 E;
varying vec3 L;
varying vec3 c;
varying float attenuation;

void main (void)
{
     float ambientAmount  = 0.0;
     float diffuseAmount  = 1.75;//2.0;
     float specularAmount = 0.25;//25;
     float specPow        = 10.00;
     vec4  matColor       = vec4(c,1);

     matColor = vec4(1,1,1,1)/2.0;
     vec3 R = reflect(-L,N);
     
     float lambertianTerm = clamp(max(dot(N,L), 0.0)  , 0.0, 1.0 );
     
     vec4 ambient = ambientAmount * matColor;     
     vec4 diffuse = attenuation * diffuseAmount * matColor * lambertianTerm; //abs(vec4(L2,1));//diffuseAmount  * clamp(matColor * max(dot(N,L), 0.0)  , 0.0, 1.0 ); 
     vec4 spec    = specularAmount * clamp ( pow(max(dot(R,E),0.0),specPow) , 0.0, 1.0 );
     gl_FragColor = clamp(ambient + diffuse + spec, 0, 1); gl_FragColor.w = 1.0;
     
     if (matColor.x == 0.0f && matColor.y == 1.0f && matColor.z == 0.0f) gl_FragColor = vec4(c,1);
}

