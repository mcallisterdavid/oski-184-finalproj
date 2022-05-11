/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The sample app's shaders.
*/

#include <metal_stdlib>
#include <simd/simd.h>

// Include header shared between this Metal shader code and C code executing Metal API commands. 
#import "ShaderTypes.h"
#import  "../Loki/loki_header.metal"

using namespace metal;

typedef struct {
    float2 position [[attribute(kVertexAttributePosition)]];
    float2 texCoord [[attribute(kVertexAttributeTexcoord)]];
} ImageVertex;

typedef struct {
    float4 position [[position]];
    float2 texCoord;
} ImageColorInOut;

struct VertexIn {
    float4 position [[ attribute(0) ]];
    float4 color [[ attribute(1) ]];
};



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
//    float dotted = -41.25229454 * rgb.r + 48.47907676 * rgb.g + 35.03935425 * rgb.b - 49.15164922 * r2 + 98.77229899 * g2 - 81.02268637 * b2 - 17.147;
    float dotted = -37.82434987000658 * rgb.r +  79.36931723128802 * rgb.g + -6.521099034515118 * rgb.b + -54.94389274556172 * r2 + 42.290986788866626 * g2 + -20.74701951531102 * b2 + -12.434318351941105;
    float greenness = 1.0 / (1.0 + exp(0.0 - dotted));
    return greenness > 0.275 ? 1.0 : 0.0;
}


// -15.2537878 ,  48.00634927,  16.87723466, -32.39931238,
//27.09634644, -41.88108443
float greenness_model_field(float4 rgb) {
    float r2 = rgb.r * rgb.r;
    float g2 = rgb.g * rgb.g;
    float b2 = rgb.b * rgb.b;
    float dotted = -15.2537878 * rgb.r + 48.00634927 * rgb.g + 16.87723466 * rgb.b - 32.39931238 * r2 + 27.09634644 * g2 - 41.88108443 * b2 - 14.46302743;
    float greenness = 1.0 / (1.0 + exp(0.0 - dotted));
    return greenness > 0.275 ? 1.0 : 0.0;
}

// 18.95133651,  -4.85966621, -10.39011094,  27.88796087,
//-7.50762479, -13.17996304

// 21.91285818,  -4.52193914,  -6.03139786,  31.65340327,
//-8.98572585, -17.39335782

float calness_model(float4 rgb) {
    float r2 = rgb.r * rgb.r;
    float g2 = rgb.g * rgb.g;
    float b2 = rgb.b * rgb.b;
    float dotted = 21.91285818 * rgb.r - 4.52193914 * rgb.g - 6.03139786 * rgb.b + 31.65340327 * r2 - 8.98572585 * g2 - 17.39335782 * b2 - 26.44;
    float greenness = 1.0 / (1.0 + exp(0.0 - dotted));
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

float whiteness_model_tbltop(float4 rgb) {
    float r2 = rgb.r * rgb.r;
    float g2 = rgb.g * rgb.g;
    float b2 = rgb.b * rgb.b;
    float dotted = 18.57690554553389 * rgb.r +  13.748518088232235 * rgb.g + 43.15077770321272 * rgb.b + -43.46284077768918 * r2 + -12.84449103882279 * g2 + 13.52908224693746 * b2 + -32.906318678767256;
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

int island_two(texture2d<float, access::read> logoTex, int x, int y) {
    int total = 0;
    for (int i = 0; i < 6; i++) {
        float curr = logoTex.read(uint2(x + i, y)).r;
        if (curr > 0.5) {
            total += 1;
        }
    }
    
    for (int i = 0; i < 6; i++) {
        float curr = logoTex.read(uint2(x - i, y)).r;
        if (curr > 0.5) {
            total += 1;
        }
    }
    
    for (int i = 0; i < 6; i++) {
        float curr = logoTex.read(uint2(x, y + i)).r;
        if (curr > 0.5) {
            total += 1;
        }
    }
    
    for (int i = 0; i < 6; i++) {
        float curr = logoTex.read(uint2(x, y - i)).r;
        if (curr > 0.5) {
            total += 1;
        }
    }
    
    return total;
}

int island(texture2d<float, access::read> logoTex, int x, int y) {
    int total = 0;
    for (int i = 0; i < 10; i++) {
        float curr = logoTex.read(uint2(x + i, y)).r;
        if (curr > 0.5) {
            total += 1;
        } else {
            break;
        }
    }
    
    for (int i = 0; i < 10; i++) {
        float curr = logoTex.read(uint2(x - i, y)).r;
        if (curr > 0.5) {
            total += 1;
        } else {
            break;
        }
    }
    
    for (int i = 0; i < 10; i++) {
        float curr = logoTex.read(uint2(x, y + i)).r;
        if (curr > 0.5) {
            total += 1;
        } else {
            break;
        }
    }
    
    for (int i = 0; i < 10; i++) {
        float curr = logoTex.read(uint2(x, y - i)).r;
        if (curr > 0.5) {
            total += 1;
        } else {
            break;
        }
    }
    
    return total;
}

//var i = 0; var numIntersections = 0; var sincePrev = 1000
//while (numIntersections < 2 && i < 400) {
//    i += 1
//    let x = max_run_index + max_run_length + Int(Double(i) * normalVec.x)
//    let y = endpointTwoY + Int(Double(i) * normalVec.y)
//    let texCoord = x + y * whiteGreenCleanTexture.width
//    if (pixelNearby(x: x, y: y, data2: data2) && sincePrev > 30) {
//        sincePrev = 0
//        numIntersections += 1
//    } else {
//        sincePrev += 1
//    }
//
//}

kernel void raycastFromLogo(texture2d<float, access::read> greenWhite [[texture(0)]],
                            const device float* inputs [[ buffer(0) ]],
                            texture2d<float, access::write> output [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
    int i = 0.0;
    int sincePrev = 1000;
    int numIntersection = 0;
    if (gid.x == 0) {
        // Left Endpoint
        while (numIntersection < 1 && i < 400) {
            i += 1;
            int x = inputs[2] + int(float(i) * inputs[8]);
            int y = inputs[3] + int(float(i) * inputs[9]);
            if (island_two(greenWhite, x, y) > 0 &&  sincePrev > 30) {
                sincePrev = 0;
                output.write(float(x + 6) / greenWhite.get_width(), uint2(4, 0));
                output.write(float(y + 6) / greenWhite.get_height(), uint2(5, 0));
                numIntersection += 1;
            } else {
                sincePrev += 1;
            }
            
        }
    } else if (gid.x == 1) {
        // Right endpoint
        while (numIntersection < 2 && i < 400) {
            i += 1;
            int x = inputs[4] + int(float(i) * inputs[6]);
            int y = inputs[5] + int(float(i) * inputs[7]);
            if (island_two(greenWhite, x, y) > 0 &&  sincePrev > 30) {
                sincePrev = 0;
                output.write(float(x + 6) / greenWhite.get_width(), uint2(numIntersection * 2, 0));
                output.write(float(y + 6) / greenWhite.get_height(), uint2(numIntersection * 2 + 1, 0));
                numIntersection += 1;
            } else {
                sincePrev += 1;
            }
            
        }
    } else if (gid.x == 2) {
//        let x = avg_x + Int(Double(i) * normalVec.x)
//        let y = avg_y + Int(Double(i) * normalVec.y)
        // Center of Mass
        while (numIntersection < 2 && i < 400) {
            i += 1;
            int x = inputs[0] + int(float(i) * inputs[6]);
            int y = inputs[1] + int(float(i) * inputs[7]);
            if (island_two(greenWhite, x, y) > 0 &&  sincePrev > 30) {
                sincePrev = 0;
                output.write(float(x + 6) / greenWhite.get_width(), uint2(6 + numIntersection * 2, 0));
                output.write(float(y + 6) / greenWhite.get_height(), uint2(6 + numIntersection * 2 + 1, 0));
                numIntersection += 1;
            } else {
                sincePrev += 1;
            }
            
        }
        
    } else {
        
    }
    
}


// Draws lines from the top of every grid
kernel void logoDetect(texture2d<float, access::read> logoTex [[texture(0)]],
                       texture2d<float, access::write> output [[texture(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
    
    bool foundFirst = false;
    uint first = 0;
    uint last = 0;
    for (uint i = 0; i < logoTex.get_height(); i++) {
        if (!foundFirst) {
            if (logoTex.read(uint2(gid.x, gid.y + i)).r > 0.5) {
                
                if (island(logoTex, gid.x, gid.y + i) > 10) {
                    first = i;
                    foundFirst = true;
                }
                
                
                
            }
        } else {
            if (logoTex.read(uint2(gid.x, gid.y + i)).r > 0.5) {
                if (i - first < 200 && island(logoTex, gid.x, gid.y + i) > 10) {
                    last = i;
                }
            }
        }
        
    }
    
    output.write(float(first) / logoTex.get_height(), uint2(gid.x, 0));
    output.write(float(last) / logoTex.get_height(), uint2(gid.x, 1));
    
//    output.write(1.0, uint2(gid.x, gid.y + first + 1));
//    output.write(1.0, uint2(gid.x, gid.y + first));
//    output.write(1.0, uint2(gid.x, gid.y + first - 1));
//    
//    output.write(1.0, uint2(gid.x, gid.y + last + 1));
//    output.write(1.0, uint2(gid.x, gid.y + last));
//    output.write(1.0, uint2(gid.x, gid.y + last - 1));
}



kernel void imageClean(texture2d<float, access::read> greenWhiteDiff [[texture(0)]],
                       texture2d<float, access::write> output [[texture(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
    float4 imageIn = greenWhiteDiff.read(gid);

    // threshold for doing random sampling
    float threshold = 0.3;

    // random number generator
    Loki rng = Loki(gid.x, gid.y, imageIn[0]);
    // radius for random sampling
    int r = 10;
    // number of random samples
    int n = 30;

    if (imageIn.r > threshold) {
        int count = 0; // how many samples are white
        for (int i = 0; i < n; i++) {
            float rng_angle = rng.rand() * 2.0;
            float rng_radius = rng.rand() * r;
            int sample_x = floor(gid.x + cospi(rng_angle) * rng_radius);
            int sample_y = floor(gid.y + sinpi(rng_angle) * rng_radius);
            float4 sample = greenWhiteDiff.read(uint2(sample_x, sample_y));
            if (sample[0] > threshold)
                count += 1;
        }
        if (count > 6) {
            output.write(1.0, gid);
        } else {
            output.write(0.0, gid);
        }

    } else {
        output.write(0.0, gid);
    }
//    output.write(imageIn[0], gid);
}

kernel void colorTransform(texture2d<float, access::read> cameraImageTextureY [[texture(0)]],
                           texture2d<float, access::read> cameraImageTextureCbCr [[texture(1)]],
                           texture2d<float, access::write> output [[texture(2)]],
                           texture2d<float, access::write> calLogoOutput [[texture(3)]],
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
    
    float sobel_x = 0.0;
    int num_white = 0;
    int num_green = 0;
    
    
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            float4 curr_y = cameraImageTextureY.read(uint2(gid.x + i, gid.y + j));
            float4 curr_cbcr = cameraImageTextureCbCr.read(uint2((gid.x + i) / 2, (gid.y + j) / 2));
            float4 curr_rgb = ycbcrToRGBTransform(curr_y, curr_cbcr);
//            float curr = greenness_model(curr_rgb);
//            curr -= whiteness_model(curr_rgb);
//            sobel_x += curr;
            num_white += whiteness_model(curr_rgb);
            num_green += greenness_model_field(curr_rgb);
        }
    }
    
    
    
    float4 rgb = ycbcrToRGBTransform(yIn, cbcrIn);
    
    
    float out_color = num_white > 0 && num_green > 0 ? 1.0 : 0.0;
    float cal_color = calness_model(rgb);
    
    float greenness = whiteness_model_tbltop(rgb);
    
//    output.write(float4(sobel_x / 9.0, 0, 0, 1), gid);
    output.write(float4(out_color, 0, 0, 1), gid);
//    output.write(float4(greenness, 0, 0, 1), gid);
    calLogoOutput.write(float4(cal_color, 0, 0, 1), gid);
//    calLogoOutput.write(float4(total / 9.0, 0, 0, 1), gid);
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

struct VertexOut {
    float4 position [[ position ]];
    float4 color;
};

vertex VertexOut baryTransform(const VertexIn vertexIn [[ stage_in ]]) {
    VertexOut vertexOut;
    vertexOut.position = vertexIn.position;
    vertexOut.color = vertexIn.color;
    
    return vertexOut;
}

fragment half4 baryFragmentShader(VertexOut vertexIn [[ stage_in ]]) {
//    return (half4(1, 0, 0, 1) + prev) / 2;
    return half4(vertexIn.color);
}

// Fog fragment function.
fragment half4 fogFragmentShader(FogColorInOut in [[ stage_in ]],
texture2d<float, access::sample> cameraImageTextureY [[ texture(0) ]],
texture2d<float, access::sample> cameraImageTextureCbCr [[ texture(1) ]],
depth2d<float, access::sample> arDepthTexture [[ texture(2) ]],
texture2d<uint> arDepthConfidence [[ texture(3) ]],
texture2d<float, access::sample> sceneTexture [[ texture(4) ]])
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
//    float edge = cameraImageTextureY.sample(s, in.texCoordCamera).r;
//
//    rgb = float4(edge, edge, edge, 1);
    float4 sceneSample = sceneTexture.sample(s, in.texCoordCamera);
    half4 cameraColor;
    if (sceneSample.r + sceneSample.g + sceneSample.b > 0) {
        cameraColor = half4(sceneSample);
    } else {
        cameraColor = half4(rgb);
    }
    

    

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

