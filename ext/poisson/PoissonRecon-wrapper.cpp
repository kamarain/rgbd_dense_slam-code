#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "Time.h"
#include "MarchingCubes.h"
#include "Octree.h"
#include "SparseMatrix.h"
#include "CmdLineParser.h"
#include "PPolynomial.h"
#include "Ply.h"
#include "MemoryUsage.h"
#include "omp.h"
#include <stdarg.h>
#include <MultiGridOctreeData.h>
/*
#include <Geometry.h>
#include <BinaryNode.h>
#include <BSplineData.h>
#include <Octree.h>
#include <SparseMatrix.h>
#include <Ply.h>
#include <Allocator.h>
#include <Array.h>
#include <MemoryUsage.h>
#include <Time.h>
//#include <PlyFile.h>
#include <MultiGridOctreeData.h>
*/
#include <PoissonRecon-wrapper.h>

char* outputFile=NULL;
int echoStdout=0;
void DumpOutput( const char* format , ... )
{
/*    if( outputFile )
    {
        FILE* fp = fopen( outputFile , "a" );
        va_list args;
        va_start( args , format );
        vfprintf( fp , format , args );
        fclose( fp );
        va_end( args );
    }
    if( echoStdout )
    {
        va_list args;
        va_start( args , format );
        vprintf( format , args );
        va_end( args );
    }*/
}
void DumpOutput2( char* str , const char* format , ... )
{/*
    if( outputFile )
    {
        FILE* fp = fopen( outputFile , "a" );
        va_list args;
        va_start( args , format );
        vfprintf( fp , format , args );
        fclose( fp );
        va_end( args );
    }
    if( echoStdout )
    {
        va_list args;
        va_start( args , format );
        vprintf( format , args );
        va_end( args );
    }
    va_list args;
    va_start( args , format );
    vsprintf( str , format , args );
    va_end( args );
    if( str[strlen(str)-1]=='\n' ) str[strlen(str)-1] = 0;
*/
}


int writePly(float *points,float *normals, int cnt, unsigned int *faceIndex, int faceCnt, int maxVerts, char *buf) {
    FILE *f = fopen(buf,"wb");
    if (f == NULL || points == NULL) return 0;
    printf("writing input data to %s\n",buf); fflush(stdout);
    fprintf(f,"ply\n");
    fprintf(f,"format ascii 1.0\n");
    fprintf(f,"element vertex %d\n",cnt);
    fprintf(f,"property float x\n");
    fprintf(f,"property float y\n");
    fprintf(f,"property float z\n");
    if (normals != NULL) {
        fprintf(f,"property float nx\n");
        fprintf(f,"property float ny\n");
        fprintf(f,"property float nz\n");
    }
    fprintf(f,"element face %d\n",faceCnt);
    fprintf(f,"property list uchar int vertex_index\n");
    fprintf(f,"end_header\n");

    if (normals != NULL) {
        for (int i = 0; i < cnt; i++) {
            fprintf(f,"%f %f %f %f %f %f\n",points[i*3+0],points[i*3+1],points[i*3+2],normals[i*3+0],normals[i*3+1],normals[i*3+2]);
        }
    } else {
        for (int i = 0; i < cnt; i++) {
            fprintf(f,"%f %f %f\n",points[i*3+0],points[i*3+1],points[i*3+2]);
        }
    }

    for (int i = 0; i < faceCnt; i++) {
        int verts = 3;
        if (maxVerts > 3) {
            verts = faceIndex[i*maxVerts+0]; fprintf(f,"%d ",verts);
            for (int j = 0; j < verts; j++) fprintf(f,"%d ",faceIndex[i*maxVerts+1+j]);
        } else {
            fprintf(f,"%d ",verts);
            for (int j = 0; j < verts; j++) fprintf(f,"%d ",faceIndex[i*maxVerts+j]);
        }
        fprintf(f,"\n");
    }
    fclose(f);
    return 1;
}
void convertPoissonMeshToRawData(CoredFileMeshData<PlyVertex<float> > &mesh, float **vertices, int *totalVerts, unsigned int **faceIndex3, int *nFaces3 ) {
    mesh.resetIterator();

    *totalVerts = int(mesh.inCorePoints.size())+mesh.outOfCorePointCount();
    *vertices   = new float[(*totalVerts)*3];
    float *vptr = *vertices;

    PlyVertex< float > p; int vi = 0;
    // incore points
    for( size_t i=0 ; i < mesh.inCorePoints.size(); i++ ) {
        p = mesh.inCorePoints[i];
        vptr[vi*3+0] = p.point[0];
        vptr[vi*3+1] = p.point[1];
        vptr[vi*3+2] = p.point[2];
        vi++;
    }
    // out-of-core points
    for( int i=0 ; i < mesh.outOfCorePointCount(); i++ ) {
        mesh.nextOutOfCorePoint(p);
        vptr[vi*3+0] = p.point[0];
        vptr[vi*3+1] = p.point[1];
        vptr[vi*3+2] = p.point[2];
        vi++;
    }

    int nFaces=mesh.polygonCount();

    const int maxVertsPerPolygon = 16;
    unsigned int *faceIndex = new unsigned int[nFaces*maxVertsPerPolygon]; // max 15 vertices per face
    printf("num triangles after poisson: %d\n",nFaces); fflush(stdout);
    std::vector< CoredVertexIndex > polygon;
    int fi = 0;
    // store all polygon data with valid index, also make sure polygons have < 16 vertex indices! >= 16 is never occurring in marching cubes
    for(int  i=0 ; i < nFaces ; i++ ) {
        mesh.nextPolygon( polygon );
        int verts = int( polygon.size() );
        // valid range: 3-15 vertices per face
        if (verts >= maxVertsPerPolygon || verts < 3) continue;
        faceIndex[fi*maxVertsPerPolygon+0] = verts;
        for( int j = 0; j < verts; j++ ) {
            if ( polygon[j].inCore) faceIndex[fi*maxVertsPerPolygon+j+1] = polygon[j].idx;
            else                    faceIndex[fi*maxVertsPerPolygon+j+1] = polygon[j].idx + int( mesh.inCorePoints.size());
        }
        fi++;
    }
    nFaces = fi;

   // writePly(vptr, NULL, *totalVerts, faceIndex, nFaces, maxVertsPerPolygon, "scratch/poisson.ply");

    // how many triangles after tesselation?
    unsigned int *facePtr3 = new unsigned int[nFaces*(maxVertsPerPolygon-2)*3]; // max 3 vertices per face
    printf("num indices after tesselation: %d\n",nFaces*(maxVertsPerPolygon-2)*3); fflush(stdout);
    *faceIndex3 = facePtr3;
    fi = 0;
    for (int i = 0; i < nFaces; i++ ) {
        int verts = faceIndex[i*maxVertsPerPolygon+0];
        // store the first triangle as is:
        facePtr3[fi*3+0] = faceIndex[i*maxVertsPerPolygon+1];
        facePtr3[fi*3+1] = faceIndex[i*maxVertsPerPolygon+2];
        facePtr3[fi*3+2] = faceIndex[i*maxVertsPerPolygon+3];
        fi++;
        // tesselate the rest:
        for (int j = 3; j < verts; j++,fi++) {
            // store the first triangle as is:
            facePtr3[fi*3+0] = facePtr3[(fi-1)*3+0];
            facePtr3[fi*3+1] = facePtr3[(fi-1)*3+2];
            facePtr3[fi*3+2] = faceIndex[i*maxVertsPerPolygon+1+j];
        }
    }
    *nFaces3 = fi;
    printf("num triangles after tesselation: %d\n",*nFaces3); fflush(stdout);
    delete[] faceIndex;
 //   writePly(vptr, NULL, *totalVerts, facePtr3, *nFaces3, 3, "scratch/poisson3.ply");
}



int writeBnpts(float *points,float *normals, int cnt, char *buf) {
    FILE *f = fopen(buf,"wb");
    if (f == NULL || points == NULL || normals == NULL) return 0;
    printf("writing input data to %s\n",buf); fflush(stdout);
    float *data = new float[cnt*6];
    for (int i = 0; i < cnt; i++) {
        data[i*6+0] = points[i*3+0];
        data[i*6+1] = points[i*3+1];
        data[i*6+2] = points[i*3+2];
        data[i*6+3] = normals[i*3+0];
        data[i*6+4] = normals[i*3+1];
        data[i*6+5] = normals[i*3+2];
    }
    fwrite(data,sizeof(float),cnt*6,f);
    delete[] data;
    fclose(f);
    return 1;
}


int generatePoissonMesh(int numThreads, int depth, float *inputPoints, float *inputNormals, int inputCount, float **vertices, int *nTotalVerts, unsigned int **faceIndex3, int *nFaces3) {
    Octree<2,false> tree;
    tree.threads = numThreads;
    int maxSolveTreeDepth = depth; // max voxel grid size := 2^n
    int minSolveTreeDepth = depth-3; // min octree subdivision level
    // The depth at which a block Gauss-Seidel solver is used
    int treeDepth = depth-1;
    // isoDivideDepth specifies the depth at which a block iso-surface extractor should be used to extract the iso-surface.
    // Using this parameter helps reduce the memory overhead at the cost of a small increase in extraction time.
    // In practice, we have found that for reconstructions of depth 9 or higher a subdivide depth of 7 or 8 can greatly reduce the memory usage.
    int isoDivideDepth = depth-1;
    // This floating point value specifies the minimum number of sample points that should fall within an octree node as the octree construction is adapted to sampling density.
    // For noise-free samples, small values in the range [1.0 - 5.0] can be used. For more noisy samples, larger values in the range [15.0 - 20.0] may be needed to provide a smoother,
    // noise-reduced, reconstruction. The default value is 1.0.
    Real samplesPerNode = 1;//4;
    // Specifies the factor of the bounding cube that the input samples should fit into.
    float scale = 1.1f;
    // Enabling this flag tells the reconstructor to use the size of the normals as confidence information.
    // When the flag is not enabled, all normals are normalized to have unit-length prior to reconstruction.
    bool useConfidence=false;//true;
    // This floating point value specifies the importants that interpolation of the point samples is given in the formulation of the screened Poisson equation.
    // The results of the original (unscreened) Poisson Reconstruction can be obtained by setting this value to 0.
    // The default value for this parameter is 4.
    bool  useNormalWeights = true;//false;
    float constraintWeight = 0.0f;

    // This specifies the exponent scale for the adaptive weighting.
    float adaptiveExponent = 1.0f;
    int minIters = 24;
    float solverAccuracy = float(1e-6);
    int fixedIters = -1;
    XForm4x4<Real> xForm = XForm4x4< Real >::Identity();
    XForm4x4<Real> iXForm = xForm.inverse();

    if( treeDepth < minSolveTreeDepth )
    {
        fprintf( stderr , "[WARNING] subdivision level must be at least as large as %d\n" , minSolveTreeDepth );
        treeDepth = minSolveTreeDepth;
    }
    if( isoDivideDepth < minSolveTreeDepth )
    {
        fprintf( stderr , "[WARNING] isodivision value must be at least as large as %d\n" , isoDivideDepth );
        isoDivideDepth = minSolveTreeDepth;
    }
    OctNode< TreeNodeData< false > , Real >::SetAllocator( MEMORY_ALLOCATOR_BLOCK_SIZE );
    //TreeOctNode::SetAllocator( MEMORY_ALLOCATOR_BLOCK_SIZE );
    //SetAllocator( MEMORY_ALLOCATOR_BLOCK_SIZE );

    int kernelDepth = depth;

    tree.setBSplineData( depth , 1 );
    if( kernelDepth>depth )
    {
        fprintf( stderr,"[ERROR] kernelDepth can't be greater than %d\n" , maxSolveTreeDepth );
        delete[] inputPoints; delete[] inputNormals;
        return 0;
    }


    //    setTree( char* fileName , int maxDepth , int minDepth , int kernelDepth , Real samplesPerNode ,
      //      Real scaleFactor , bool useConfidence , bool useNormalWeights , Real constraintWeight , int adaptiveExponent , XForm4x4< Real > xForm=XForm4x4< Real >::Identity );

    // store visualization into given buffer
    // generateVisualization(points,normals,cnt,vbuf);
    char buf[512]; sprintf(buf,"scratch/tempmesh.bnpts");
    if (!writeBnpts(inputPoints,inputNormals,inputCount,buf)) {
        printf("writing data to file %s failed!\n",buf);
        delete[] inputPoints; delete[] inputNormals;
        return 0;
    }
    delete[] inputPoints; delete[] inputNormals;

    int pointCount = tree.setTree(&buf[0],maxSolveTreeDepth , minSolveTreeDepth , kernelDepth , samplesPerNode , scale, useConfidence, useNormalWeights, constraintWeight,adaptiveExponent,xForm);
    printf("tree.setTree returns %d points\n",pointCount); fflush(stdout);

    tree.ClipTree();
    tree.finalize( isoDivideDepth );

    printf( "Input Points: %d\n" , pointCount );
    printf( "Leaves/Nodes: %d/%d\n" , tree.tree.leaves() , tree.tree.nodes() );
    printf( "Memory Usage: %.3f MB\n" , float( MemoryInfo::Usage() )/(1<<20) );

    if (pointCount > 1000) {
        int maxMemoryUsage = tree.maxMemoryUsage;
        tree.maxMemoryUsage=0;
        tree.SetLaplacianConstraints();
        printf( "Memory Usage after Laplacian: %.3f MB\n" , float( MemoryInfo::Usage())/(1<<20) );
        maxMemoryUsage = std::max< double >( maxMemoryUsage , tree.maxMemoryUsage );

        tree.maxMemoryUsage=0;
        tree.LaplacianMatrixIteration( treeDepth, false , minIters, solverAccuracy, maxSolveTreeDepth , fixedIters);
        printf( "Memory Usage: %.3f MB\n" , float( MemoryInfo::Usage() )/(1<<20) );
        maxMemoryUsage = std::max< double >( maxMemoryUsage , tree.maxMemoryUsage );

        CoredFileMeshData<PlyVertex< float > > mesh;
        tree.maxMemoryUsage=0;
        Real isoValue = tree.GetIsoValue();
        printf( "Iso-Value: %e\n" , isoValue );

        tree.maxMemoryUsage = 0;
        tree.GetMCIsoTriangles( isoValue , isoDivideDepth , &mesh , 0 , 1 , false , true );
        maxMemoryUsage = std::max< double >( maxMemoryUsage , tree.maxMemoryUsage );

        convertPoissonMeshToRawData(mesh, vertices, nTotalVerts, faceIndex3, nFaces3);
        return 1;
    }
    return 0;
}
