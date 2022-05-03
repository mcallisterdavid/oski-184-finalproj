/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The sample app's shaders.
*/

#include <metal_stdlib>
#include <simd/simd.h>

// Include header shared between this Metal shader code and C code executing Metal API commands. 
#import "ShaderTypes.h"

using namespace metal;

typedef struct {
    float2 position [[attribute(kVertexAttributePosition)]];
    float2 texCoord [[attribute(kVertexAttributeTexcoord)]];
} ImageVertex;

typedef struct {
    float4 position [[position]];
    float2 texCoord;
} ImageColorInOut;



// Convert from YCbCr to rgb.
float4 ycbcrToRGBTransform(float4 y, float4 CbCr) {
    const float4x4 ycbcrToRGBTransform = float4x4(
      float4(+1.0000f, +1.0000f, +1.0000f, +0.0000f),
      float4(+0.0000f, -0.3441f, +1.7720f, +0.0000f),
      float4(+1.4020f, -0.7141f, +0.0000f, +0.0000f),
      float4(-0.7010f, +0.5291f, -0.8860f, +1.0000f)
    );

    float4 ycbcr = float4(y.r, CbCr.rg, 1.0);
    return ycbcrToRGBTransform * ycbcr;
}

float manhattanDistance(float4 truth, float4 test) {
    return 0.5 * abs(truth.r - test.r) + 2.0 * abs(truth.g - test.g) + 0.5 * abs(truth.b - test.b);
}

float greenness_model(float4 rgb) {
    float r2 = rgb.r * rgb.r;
    float g2 = rgb.g * rgb.g;
    float b2 = rgb.b * rgb.b;
    float dotted = -41.25229454 * rgb.r + 48.47907676 * rgb.g + 35.03935425 * rgb.b - 49.15164922 * r2 + 98.77229899 * g2 - 81.02268637 * b2 - 17.147;
    float greenness = 1.0 / (1.0 + exp(0.0 - dotted));
//    if (greenness > 0.275) {
//        greenness = 1;
//    } else {
//        greenness = 0;
//    }
    return greenness > 0.275 ? 1.0 : 0.0;
}

float whiteness_model(float4 rgb) {
    float r2 = rgb.r * rgb.r;
    float g2 = rgb.g * rgb.g;
    float b2 = rgb.b * rgb.b;
    float dotted = -8.67260227 * rgb.r - 31.35452913 * rgb.g + 38.64 * rgb.b - 7.57657822 * r2 - 42.85956885 * g2 + 65.68466432 * b2 - 3.035;
    float whiteness = 1.0 / (1.0 + exp(0.0 - dotted));
    whiteness -= 0.45;
    whiteness *= 2;
    return whiteness > 0.4 ? 1.0 : 0.0;
}

typedef struct {
    float2 position;
    float2 texCoord;
} FogVertex;

typedef struct {
    float4 position [[position]];
    float2 texCoordCamera;
    float2 texCoordScene;
} FogColorInOut;

kernel void colorTransform(texture2d<float, access::read> cameraImageTextureY [[texture(0)]],
                           texture2d<float, access::read> cameraImageTextureCbCr [[texture(1)]],
                           texture2d<float, access::write> output [[texture(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    float4 yIn = cameraImageTextureY.read(gid);
    // yIn.r -> luma
    float4 cbcrIn = cameraImageTextureCbCr.read(uint2(gid.x/2, gid.y/2));
    // cbcrIn.r -> cb; cbcr.In.g -> cr
    
//    for (int i = -1; i < 2; i++) {
//        for (int j = -1; j < 2; j++) {
//            yAvg += cameraImageTextureY.read(uint2(gid.x + i, gid.y + j)) / 9.0;
//        }
//    }
    
    const float3x3 sobel_kernel_x = float3x3(
    float3(1.0f, 0.0f, -1.0f),
    float3(2.0f, 0.0f, -2.0f),
    float3(1.0f, 0.0f, -1.0f));
    
    
    
    const float3x3 sobel_kernel_y = float3x3(
    float3(1.0, 2.0, 1.0),
    float3(0, 0, 0),
    float3(-1.0, -2.0, -1.0));
    
    float sobel_x = 0.0;
    
//    float4 curr_y;
//    float4 curr_cbcr;
//    float4 curr_rgb;
//    float curr;
//
//    curr_y = cameraImageTextureY.read(uint2(gid.x - 1, gid.y));
//    curr_cbcr = cameraImageTextureCbCr.read(uint2((gid.x - 1) / 2, (gid.y) / 2));
//    curr_rgb = ycbcrToRGBTransform(curr_y, curr_cbcr);
//    curr = greenness_model(curr_rgb);
//    curr -= whiteness_model(curr_rgb);
//    sobel_x += curr * 2.0;
//
//    curr_y = cameraImageTextureY.read(uint2(gid.x + 1, gid.y));
//    curr_cbcr = cameraImageTextureCbCr.read(uint2((gid.x + 1) / 2, (gid.y) / 2));
//    curr_rgb = ycbcrToRGBTransform(curr_y, curr_cbcr);
//    curr = greenness_model(curr_rgb);
//    curr -= whiteness_model(curr_rgb);
//    sobel_x -= curr * 2.0;
//
//    curr_y = cameraImageTextureY.read(uint2(gid.x - 1, gid.y + 1));
//    curr_cbcr = cameraImageTextureCbCr.read(uint2((gid.x - 1) / 2, (gid.y + 1) / 2));
//    curr_rgb = ycbcrToRGBTransform(curr_y, curr_cbcr);
//    curr = greenness_model(curr_rgb);
//    curr -= whiteness_model(curr_rgb);
//    sobel_x += curr * 1.0;
//
//    curr_y = cameraImageTextureY.read(uint2(gid.x + 1, gid.y + 1));
//    curr_cbcr = cameraImageTextureCbCr.read(uint2((gid.x + 1) / 2, (gid.y + 1) / 2));
//    curr_rgb = ycbcrToRGBTransform(curr_y, curr_cbcr);
//    curr = greenness_model(curr_rgb);
//    curr -= whiteness_model(curr_rgb);
//    sobel_x -= curr * 1.0;
//
//    curr_y = cameraImageTextureY.read(uint2(gid.x - 1, gid.y - 1));
//    curr_cbcr = cameraImageTextureCbCr.read(uint2((gid.x - 1) / 2, (gid.y - 1) / 2));
//    curr_rgb = ycbcrToRGBTransform(curr_y, curr_cbcr);
//    curr = greenness_model(curr_rgb);
//    curr -= whiteness_model(curr_rgb);
//    sobel_x += curr * 1.0;
//
//    curr_y = cameraImageTextureY.read(uint2(gid.x + 1, gid.y - 1));
//    curr_cbcr = cameraImageTextureCbCr.read(uint2((gid.x + 1) / 2, (gid.y - 1) / 2));
//    curr_rgb = ycbcrToRGBTransform(curr_y, curr_cbcr);
//    curr = greenness_model(curr_rgb);
//    curr -= whiteness_model(curr_rgb);
//    sobel_x -= curr * 1.0;
    
    
    
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            float4 curr_y = cameraImageTextureY.read(uint2(gid.x + i, gid.y + j));
            float4 curr_cbcr = cameraImageTextureCbCr.read(uint2((gid.x + i) / 2, (gid.y + j) / 2));
            float4 curr_rgb = ycbcrToRGBTransform(curr_y, curr_cbcr);
            float curr = greenness_model(curr_rgb);
            curr -= whiteness_model(curr_rgb);
            sobel_x += curr;
        }
    }
    
//    float4 rgb = ycbcrToRGBTransform(
//        yIn,
//        cbcrIn);
//
//
//    bool greenness = greenness_model(rgb);
//    bool whiteness = whiteness_model(rgb);
    
    output.write(float4(sobel_x / 9.0, 0, 0, 1), gid);
}

// Fog the image vertex function.
vertex FogColorInOut fogVertexTransform(const device FogVertex* cameraVertices [[ buffer(0) ]],
                                                         const device FogVertex* sceneVertices [[ buffer(1) ]],
                                                         unsigned int vid [[ vertex_id ]]) {
    FogColorInOut out;

    const device FogVertex& cv = cameraVertices[vid];
    const device FogVertex& sv = sceneVertices[vid];

    out.position = float4(cv.position, 0.0, 1.0);
    out.texCoordCamera = cv.texCoord;
    out.texCoordScene = sv.texCoord;

    return out;
}

// Fog fragment function.
fragment half4 fogFragmentShader(FogColorInOut in [[ stage_in ]],
texture2d<float, access::sample> cameraImageTextureY [[ texture(0) ]],
texture2d<float, access::sample> cameraImageTextureCbCr [[ texture(1) ]],
depth2d<float, access::sample> arDepthTexture [[ texture(2) ]],
texture2d<uint> arDepthConfidence [[ texture(3) ]],
texture2d<float, access::write> lineDraw [[ texture(4) ]])
{
    // Whether to show the confidence debug visualization.
    // - Tag: ConfidenceVisualization
    // Set to `true` to visualize confidence.
    bool confidenceDebugVisualizationEnabled = false;
    
    // Set the maximum fog saturation to 4.0 meters. Device maximum is 5.0 meters.
    const float fogMax = 4.0;
    
    // Fog is fully opaque, middle grey
    const half4 fogColor = half4(0.5, 0.5, 0.5, 1.0);
    
    // Confidence debug visualization is red.
    const half4 confidenceColor = half4(1.0, 0.0, 0.0, 1.0);
    
    // Maximum confidence is `ARConfidenceLevelHigh` = 2.
    const uint maxConfidence = 2;
    
    // Create an object to sample textures.
    constexpr sampler s(address::clamp_to_edge, filter::linear);

    // Sample this pixel's camera image color.
    float4 rgb = ycbcrToRGBTransform(
        cameraImageTextureY.sample(s, in.texCoordCamera),
        cameraImageTextureCbCr.sample(s, in.texCoordCamera)
    );
//    rgb = cameraImageTextureY.sample(s, in.texCoordCamera) - cameraImageTextureCbCr.sample(s, in.texCoordCamera);
    half4 cameraColor = half4(rgb);

    

    // Sample this pixel's depth value.
    float depth = arDepthTexture.sample(s, in.texCoordCamera);
    
    // Ignore depth values greater than the maximum fog distance.
    depth = clamp(depth, 0.0, fogMax);
    
    // Determine this fragment's percentage of fog.
    float fogPercentage = depth / fogMax;
    
    // Mix the camera and fog colors based on the fog percentage.
    half4 foggedColor = mix(cameraColor, cameraColor, fogPercentage);
    
    
    // Just return the fogged color if confidence visualization is disabled.
    if(!confidenceDebugVisualizationEnabled) {
        return foggedColor;
    } else {
        // Sample the depth confidence.
        uint confidence = arDepthConfidence.sample(s, in.texCoordCamera).x;
        
        // Assign a color percentage based on confidence.
        float confidencePercentage = (float)confidence / (float)maxConfidence;

        // Return the mixed confidence and foggedColor.
        return mix(confidenceColor, foggedColor, confidencePercentage);
    }
}

