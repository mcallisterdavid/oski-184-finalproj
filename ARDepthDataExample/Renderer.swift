/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The host app renderer.
*/

import Foundation
import Metal
import MetalKit
import ARKit
import MetalPerformanceShaders

protocol RenderDestinationProvider {
    var currentRenderPassDescriptor: MTLRenderPassDescriptor? { get }
    var currentDrawable: CAMetalDrawable? { get }
    var colorPixelFormat: MTLPixelFormat { get set }
    var sampleCount: Int { get set }
}

struct Vertex {
    var position: SIMD3<Float>
    var color: SIMD4<Float>
}

// The max number of command buffers in flight.
let kMaxBuffersInFlight: Int = 3

var numFrames: Int = 0
var frameProcessed = true

let thetaRes = 1
let rhoRes = 5

// Vertex data for an image plane.
let kImagePlaneVertexData: [Float] = [
    -1.0, -1.0, 0.0, 1.0,
    1.0, -1.0, 1.0, 1.0,
    -1.0, 1.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 0.0
]

class Renderer {
    let session: ARSession
    let sceneRenderer: SCNRenderer
    let device: MTLDevice
    let inFlightSemaphore = DispatchSemaphore(value: kMaxBuffersInFlight)
    var renderDestination: RenderDestinationProvider
    let numLines: Int = 5
    var lineUpdate:Int = 0
    
    var lastThree: [[Int]] = []
    
    // Metal objects.
    var commandQueue: MTLCommandQueue!
    
    var lines: [Array<Int>] = []
    var newLines: [Line] = []
    
    var baryVertices: [Vertex] = []
    
    // An object that holds vertex information for source and destination rendering.
    var imagePlaneVertexBuffer: MTLBuffer!
    
    // An object that defines the Metal shaders that render the camera image and fog.
    var fogPipelineState: MTLRenderPipelineState!
    
    var baryPipelineState: MTLRenderPipelineState!
    
//    var tilePipelineState: MTLRenderPipelineState!
    
    var computePipelineState : MTLComputePipelineState!;
    
    var logoRaycastComputePipelineState: MTLComputePipelineState!;
    
    var cleanComputePipelineState : MTLComputePipelineState!;
    
    var greenWhiteCleanComputePipelineState : MTLComputePipelineState!;
    
    var logoComputePipelineState : MTLComputePipelineState!;

    // Textures used to transfer the current camera image to the GPU for rendering.
    var cameraImageTextureY: CVMetalTexture?
    var cameraImageTextureCbCr: CVMetalTexture?
    
    // A texture used to store depth information from the current frame.
    var depthTexture: CVMetalTexture?
    
    // A texture used to pass confidence information to the GPU for fog rendering.
    var confidenceTexture: CVMetalTexture?
    
    // A texture of the blurred depth data to pass to the GPU for fog rendering.
    var filteredDepthTexture: MTLTexture!
    
    var filteredYTexture: MTLTexture!
    
    var tileOutTexture: MTLTexture!
    
    var cleanOutTexture: MTLTexture!
    
    var whiteGreenCleanTexture: MTLTexture!
    
    var calLogoTexture: MTLTexture!
    
    var logoRaycastOutTexture: MTLTexture!
    
    var logoRaycastTexture: MTLTexture!
    
    var logoStart: [Int]!
    
    var logoCoM: [Int]!
    
    var ogTransform: SCNMatrix4!
    
    var sceneTexture: MTLTexture!
    
    // Texture of height 2 storing the top and bottom y values for each vertical strip of the callogotexture
    var topBottomTexture: MTLTexture!
    
    var lineTexture: MTLTexture!
    
    // A filter used to blur the depth data for rendering fog.
    var downsizeFilter: MPSImageLanczosScale?
    
    // A filter used to blur the depth data for rendering fog.
    var sobelFilter: MPSImageSobel?
    
    // Captured image texture cache.
    var cameraImageTextureCache: CVMetalTextureCache!
    
    // The current viewport size.
    var viewportSize: CGSize = CGSize()
    
    // Flag for viewport size changes.
    var viewportSizeDidChange: Bool = false
    
    // Initialize a renderer by setting up the AR session, GPU, and screen backing-store.
    init(session: ARSession, metalDevice device: MTLDevice, renderDestination: RenderDestinationProvider, scnRenderer: SCNRenderer) {
        self.session = session
        self.device = device
        self.renderDestination = renderDestination
        self.sceneRenderer = scnRenderer
        
        
        // Perform one-time setup of the Metal objects.
        loadMetal()
    }
    
    // Schedule a draw to happen at a new size.
    func drawRectResized(size: CGSize) {
        viewportSize = size
        viewportSizeDidChange = true
    }
    
    
    func update() {
        // Wait to ensure only kMaxBuffersInFlight are getting proccessed by any stage in the Metal
        // pipeline (App, Metal, Drivers, GPU, etc).
        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        
        
        // Create a new command buffer for each renderpass to the current drawable.
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            commandBuffer.label = "MyCommand"
            
            
            
            // Add completion hander which signal _inFlightSemaphore when Metal and the GPU has fully
            // finished proccssing the commands we're encoding this frame.  This indicates when the
            // dynamic buffers, that we're writing to this frame, will no longer be needed by Metal
            // and the GPU.
            commandBuffer.addCompletedHandler { [weak self] commandBuffer in
                if let strongSelf = self {
                    strongSelf.inFlightSemaphore.signal()
                }
            }
            
            updateAppState()
            
            
//            applySobel(commandBuffer: commandBuffer)
//            applyGaussianBlur(commandBuffer: commandBuffer)
            
            if let renderPassDescriptor = renderDestination.currentRenderPassDescriptor, let currentDrawable = renderDestination.currentDrawable, let offscreenTexture = sceneTexture {
                
                renderPassDescriptor.colorAttachments[0].texture = offscreenTexture
                renderPassDescriptor.colorAttachments[0].loadAction = .clear
                renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0.0); //green
                renderPassDescriptor.colorAttachments[0].storeAction = .store
                
//                sceneRenderer.scene!.rootNode.camera?.projectionTransform = SCNMatrix4(session.currentFrame?.camera.intrinsics)
                
                if (sceneRenderer.pointOfView != nil && self.logoCoM != nil) {
                
                    if (self.ogTransform == nil) {
                        self.ogTransform = sceneRenderer.pointOfView!.transform
                    }
                    
                    let column0: SIMD4<Float> = [1, 0, 0, 0]
                    let column1: SIMD4<Float> = [0, 0.5, 0.8, 0]
                    let column2: SIMD4<Float> = [0, -0.8, 0.5, 0]
                    let column3: SIMD4<Float> = [0, -60, 53, 1]
                    let transform = simd_float4x4(columns: (column0,
                        column1,
                        column2,
                        column3))
                    
                    
                    let test_extrinsic = simd_float4x4(columns: ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [100, 0, -100, 1]))
                    
    //                let anch = ARAnchor(transform: transform)
                    
                    let camIntrins = session.currentFrame?.camera.intrinsics
                    
                    let intrinsic = simd_float4x3(columns: (camIntrins!.columns.0, camIntrins!.columns.1, camIntrins!.columns.2, [0, 0, 0]))
                    
                    print(intrinsic * test_extrinsic)
                    
                    let theta = Float(numFrames) * Float.pi / 30.0
                    var extrinsic = simd_float4x4(columns: ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [cos(theta) * 1000, sin(theta) * 1000, 0, 1]))
                    
    //                let intrinsic_two = simd_float4x4(columns: (camIntrins!.columns.0, camIntrins!.columns.1, camIntrins!.columns.2, [0, 0, 0]))
                    let mulled = intrinsic * extrinsic
                    
//                    SCNMatrix4MakeRotation(<#T##angle: Float##Float#>, <#T##x: Float##Float#>, <#T##y: Float##Float#>, <#T##z: Float##Float#>)
                    
                    
                    var x = 0
                    var y = 0
                    for pnt in self.lastThree {
                        x += pnt[0]
                        y += pnt[1]
                    }
                    x /= lastThree.count
                    y /= lastThree.count
                    
                    
                    
                    let logoDiff = [x - self.logoStart[0], y - self.logoStart[1]]
                    
                    let rotate = SCNMatrix4MakeRotation(Float(logoDiff[0]) / 2000, 0, 0.45, 1)
                    
                    let translate = SCNMatrix4Translate(ogTransform, 0.0 - Float(logoDiff[0]) / 15, 0.0 + Float(logoDiff[1]) / 30, 0)
                    
//                    sceneRenderer.pointOfView?.transform = SCNMatrix4Mult(ogTransform, SCNMatrix4MakeScale(1 + 0.3 * sin(theta), 1 + 0.3 * sin(theta), 1 + 0.3 * sin(theta)))
                    
                    sceneRenderer.pointOfView?.transform = SCNMatrix4Mult(translate, rotate)
                    
//                    sceneRenderer.pointOfView?.transform = translate
                    
                    
                    
    //                print(intrinsic * extrinsic)
                    
                    
    //                let t = SIMD4(mulled.columns.0, 0)
                    
                
                    
                    let appended = simd_float4x4(columns: (SIMD4(mulled.columns.0, 0), SIMD4(mulled.columns.1, 0), SIMD4(mulled.columns.2, 0), SIMD4(mulled.columns.0, 1)))
                
    //                print(camIntrins)
    //                print(session.currentFrame?.camera.eulerAngles)
    //                session.add(anchor: anch)
                    
    //                print(sceneRenderer.pointOfView?.transform)
    //                sceneRenderer.pointOfView?.transform = SCNMatrix4(appended)
                
                }
                
                
                
                
                self.sceneRenderer.render(atTime: CFTimeInterval(), viewport: CGRect(x: 0, y: 0, width: offscreenTexture.width, height: offscreenTexture.height), commandBuffer: commandBuffer, passDescriptor: renderPassDescriptor)
            }

            
            
            if let renderPassDescriptor = renderDestination.currentRenderPassDescriptor, let currentDrawable = renderDestination.currentDrawable {
                
                
                // Check these values
                
                if let fogRenderEncoding = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                    
                    // Set a label to identify this render pass in a captured Metal frame.
                    fogRenderEncoding.label = "MyFogRenderEncoder"
                    
                    

                    // Schedule the camera image and fog to be drawn to the screen.
                    doFogRenderPass(renderEncoder: fogRenderEncoding)
//                    doBaryPass(renderEncoder: fogRenderEncoding)
                    
                    
                    // Finish encoding commands.
                    fogRenderEncoding.endEncoding()
                    
                    
                    
                    
                }
                
                // Schedule a present once the framebuffer is complete using the current drawable.
                commandBuffer.present(currentDrawable)
                
                
            }
            
                        
            
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                computeEncoder.setComputePipelineState(computePipelineState)
                
                if let cameraImageY = cameraImageTextureY, let cameraImageCbCr = cameraImageTextureCbCr {
                    
                    let MTLTextureY = CVMetalTextureGetTexture(cameraImageY)
                    let MTLTextureCbCr = CVMetalTextureGetTexture(cameraImageCbCr)
                    
                    if (tileOutTexture == nil) {
                        setupYFilter(width: MTLTextureY!.width, height: MTLTextureY!.height)
                    }
                    
                    computeEncoder.setTexture(MTLTextureY, index: 0)
                    
                    computeEncoder.setTexture(MTLTextureCbCr, index: 1)
                    
                    let threadgroupSize = MTLSizeMake(16, 16, 1);
                    
                    var threadgroupCount = MTLSize();
                    
                    threadgroupCount.width  = (MTLTextureY!.width  + threadgroupSize.width -  1) / threadgroupSize.width;
                    threadgroupCount.height = (MTLTextureY!.height + threadgroupSize.height - 1) / threadgroupSize.height;
                    threadgroupCount.depth = 1
                    
                    let w = computePipelineState.threadExecutionWidth
                    let h = computePipelineState.maxTotalThreadsPerThreadgroup / w
                    let threadsPerThreadgroup = MTLSizeMake(w, h, 1)

                    let threadsPerGrid = MTLSize(width: MTLTextureY!.width,
                                                 height: MTLTextureY!.height,
                                                 depth: 1)
                    
                    computeEncoder.setTexture(tileOutTexture, index: 2)
                    
                    computeEncoder.setTexture(calLogoTexture, index: 3)
                    
                    computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                    
                }
                
                
                computeEncoder.endEncoding()
            }
            
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                computeEncoder.setComputePipelineState(cleanComputePipelineState)

                if let greenWhiteIn = calLogoTexture {


                    computeEncoder.setTexture(calLogoTexture, index: 0)

                    let threadgroupSize = MTLSizeMake(16, 16, 1);

                    var threadgroupCount = MTLSize();

                    threadgroupCount.width  = (calLogoTexture.width  + threadgroupSize.width -  1) / threadgroupSize.width;
                    threadgroupCount.height = (calLogoTexture.height + threadgroupSize.height - 1) / threadgroupSize.height;
                    threadgroupCount.depth = 1

                    let w = cleanComputePipelineState.threadExecutionWidth
                    let h = cleanComputePipelineState.maxTotalThreadsPerThreadgroup / w
                    let threadsPerThreadgroup = MTLSizeMake(w, h, 1)

                    let threadsPerGrid = MTLSize(width: calLogoTexture.width,
                                                 height: calLogoTexture.height,
                                                 depth: 1)

                    computeEncoder.setTexture(cleanOutTexture, index: 1)

                    computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

                }

                computeEncoder.endEncoding()
            }
            
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                computeEncoder.setComputePipelineState(greenWhiteCleanComputePipelineState)

                if let greenWhiteIn = tileOutTexture {


                    computeEncoder.setTexture(greenWhiteIn, index: 0)

                    let threadgroupSize = MTLSizeMake(16, 16, 1);

                    var threadgroupCount = MTLSize();

                    threadgroupCount.width  = (greenWhiteIn.width  + threadgroupSize.width -  1) / threadgroupSize.width;
                    threadgroupCount.height = (greenWhiteIn.height + threadgroupSize.height - 1) / threadgroupSize.height;
                    threadgroupCount.depth = 1

                    let w = greenWhiteCleanComputePipelineState.threadExecutionWidth
                    let h = greenWhiteCleanComputePipelineState.maxTotalThreadsPerThreadgroup / w
                    let threadsPerThreadgroup = MTLSizeMake(w, h, 1)

                    let threadsPerGrid = MTLSize(width: greenWhiteIn.width,
                                                 height: greenWhiteIn.height,
                                                 depth: 1)

                    computeEncoder.setTexture(whiteGreenCleanTexture, index: 1)

                    computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

                }

                computeEncoder.endEncoding()
            }
            
            if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                computeEncoder.setComputePipelineState(logoComputePipelineState)

                if let logoTex = cleanOutTexture {


                    computeEncoder.setTexture(cleanOutTexture, index: 0)

                    let threadgroupSize = MTLSizeMake(16, 16, 1);

                    var threadgroupCount = MTLSize();

                    threadgroupCount.width  = (logoTex.width  + threadgroupSize.width -  1) / threadgroupSize.width;
                    threadgroupCount.height = 1;
                    threadgroupCount.depth = 1

                    let w = logoComputePipelineState.threadExecutionWidth
                    let h = logoComputePipelineState.maxTotalThreadsPerThreadgroup / w
                    let threadsPerThreadgroup = MTLSizeMake(w, h, 1)

                    let threadsPerGrid = MTLSize(width: cleanOutTexture.width,
                                                 height: 1,
                                                 depth: 1)

                    computeEncoder.setTexture(topBottomTexture, index: 1)

                    computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                    
                    

                }
                computeEncoder.endEncoding()
            }
            
            var inputs: [Float] = []
                
                
            if let greenWhite = self.tileOutTexture, let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
                    computeEncoder.setComputePipelineState(logoRaycastComputePipelineState)

                inputs = self.fieldLogoDetection(width: self.calLogoTexture.width, height: self.calLogoTexture.height)
                
                if inputs.count > 0 {

                    computeEncoder.setTexture(greenWhite, index: 0)

                    let threadgroupSize = MTLSizeMake(16, 16, 1);

                    var threadgroupCount = MTLSize();

                    threadgroupCount.width  = 3;
                    threadgroupCount.height = 1;
                    threadgroupCount.depth = 1

                    let w = logoRaycastComputePipelineState.threadExecutionWidth
                    let h = logoRaycastComputePipelineState.maxTotalThreadsPerThreadgroup / w
                    let threadsPerThreadgroup = MTLSizeMake(w, h, 1)

                    let threadsPerGrid = MTLSize(width: 3,
                                                 height: 1,
                                                 depth: 1)
                    
                                
                    let inputBuffer = device.makeBuffer(bytes: inputs, length: inputs.count * MemoryLayout<Float>.size, options: [])
                                
                    computeEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
                    
                    computeEncoder.setTexture(logoRaycastOutTexture, index: 1)

                    computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

                }

                computeEncoder.endEncoding()
            }

                
            
            
//            applySobel(commandBuffer: commandBuffer)
            
            // Finalize rendering here & push the command buffer to the GPU.
            commandBuffer.commit()
            
            commandBuffer.addCompletedHandler({_ in
                if (self.cleanOutTexture != nil && frameProcessed) {
                    frameProcessed = false
//                    Task.init{await self.readTileOut()}
//                    Task.init{await self.findLines()}

                }
                if (self.whiteGreenCleanTexture != nil && inputs.count > 0) {
                    self.readTileOut(width: self.whiteGreenCleanTexture.width, height: self.whiteGreenCleanTexture.height, CoMX: inputs[0], CoMY: inputs[1])
                }
//                if (self.calLogoTexture != nil) {
//                    self.fieldLogoDetection(width: self.calLogoTexture.width, height: self.calLogoTexture.height)
//                }
                numFrames += 1
            })
            
            
            
            
            
        }
    }
   
    
    
    
    
    func readTileOut(width: Int, height: Int, CoMX: Float, CoMY: Float) {
        if (logoRaycastOutTexture != nil) {
            let tex = logoRaycastOutTexture!
            
            let bytesPerPixel = 2
            
            let bytesPerRow = tex.width * bytesPerPixel
            
            var data = [UInt16](repeating: 0, count: tex.width*tex.height)

            
            
            tex.getBytes(&data,
                bytesPerRow: bytesPerRow,
                from: MTLRegionMake2D(0, 0, tex.width, tex.height),
                mipmapLevel: 0)
        
            
            let m = data.max()!
            
            if (m < 10) {
                return
            }
            
            let m0 = simd_float2(CoMX, CoMY)
            let X0 = simd_float3(0, 0, 0)
            
            let pt1: [Int] = [Int(Double(data[0]) / Double(UInt16.max) * Double(width)), Int(Double(data[1]) / Double(UInt16.max) * Double(height))]
            addMetalTriangleVerts(x: pt1[0], y: pt1[1])
            
            let m1 = simd_float2(xImageSpacetoNormalized(x: pt1[0]), yImageSpacetoNormalized(y: pt1[1]))
            let X1 = simd_float3(5, -18, 0)
            
            // Right tick mark
            let pt2: [Int] = [Int(Double(data[2]) / Double(UInt16.max) * Double(width)), Int(Double(data[3]) / Double(UInt16.max) * Double(height))]
            addMetalTriangleVerts(x: pt2[0], y: pt2[1])
            
            let m2 = simd_float2(xImageSpacetoNormalized(x: pt2[0]), yImageSpacetoNormalized(y: pt2[1]))
            let X2 = simd_float3(4, -23, 0)
            
            let pt3: [Int] = [Int(Double(data[4]) / Double(UInt16.max) * Double(width)), Int(Double(data[5]) / Double(UInt16.max) * Double(height))]
            addMetalTriangleVerts(x: pt3[0], y: pt3[1])
            
            let m3 = simd_float2(xImageSpacetoNormalized(x: pt3[0]), yImageSpacetoNormalized(y: pt3[1]))
            let X3 = simd_float3(-5, -24, 0)
            
            
            let pt4: [Int] = [Int(Double(data[6]) / Double(UInt16.max) * Double(width)), Int(Double(data[7]) / Double(UInt16.max) * Double(height))]
            addMetalTriangleVerts(x: pt4[0], y: pt4[1])
            let pt5: [Int] = [Int(Double(data[8]) / Double(UInt16.max) * Double(width)), Int(Double(data[9]) / Double(UInt16.max) * Double(height))]
            addMetalTriangleVerts(x: pt5[0], y: pt5[1])
            
            
            // Perform algorithm
            
            
            // Step 1
            
            let n_x = (X1 - X0) / sqrt(dot((X1 - X0), (X1 - X0)))
            let x3x1 = X2 - X0
            let inter = cross(n_x, x3x1)
            let n_z = inter / sqrt(dot(inter, inter))
            let n_y = cross(n_z, n_x)
            
            let N = simd_float3x3(columns: (n_x, n_y, n_z))
            
            let X_p_0 = N.transpose * (X0 - X0)
            let X_p_1 = N.transpose * (X1 - X0)
            let X_p_2 = N.transpose * (X2 - X0)
            
            // Step 2
            
            let a = simd_length(X1 - X0)
            let c: Float = 0.1
            let b = sqrt(dot((X2 - X0), (X2 - X0)) - c*c)
            
            let p = b/a
            let q = (b*b + c*c) / (a*a)
            
            let A = simd_float3x2(columns: (-m0, m1, simd_float2.zero))
            let B = simd_float3x2(columns: (-m0, simd_float2.zero, m2))
            let C = B - p*A
            
            let f_1 = p
            let f_2 = dot(m1, m2)
            let f_4 = (1 - 2*p) * dot(m0, m1)
            let f_5 = dot(m0, m2)
            let f_6 = 1 - p
            
            let g_1 = q
            let g_3: Float = -1
            let g_4 = -2 * q * dot(m0, m1)
            let g_5 = 2 * f_5
            let g_6 = q - 1
            
            let h_1 = f_2 * f_2 * g_1 - f_1 * f_1
            let h_2 = f_2 * f_2 * g_4 + 2 * f_2 * f_5 * (g_1 - f_1) - 2 * f_1 * f_4
            let h_3 = (f_5 * f_5) * (g_1 - 2 * f_1) + (2 * f_2 * f_5) * (g_4 - f_4) - (2 * f_1 * f_6) + (f_2 * f_2 * g_6) - (f_4 * f_4)
            let h_4 = f_5 * f_5 * (g_4 - 2 * f_4) + 2 * f_2 * f_5 * (g_6 - f_6) - 2 * f_4 * f_6
            let h_5 = f_5 * f_5 * (g_6 - 2 * f_6) - f_6 * f_6
            
            
            
            
            // CUTOFF
            
//            var data2 = [UInt8](repeating: 0, count: whiteGreenCleanTexture!.width*whiteGreenCleanTexture!.height)
//            self.whiteGreenCleanTexture!.getBytes(&data2, bytesPerRow: whiteGreenCleanTexture!.width, from: MTLRegionMake2D(0, 0, whiteGreenCleanTexture!.width, whiteGreenCleanTexture!.height), mipmapLevel: 0)
//
//            let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
//            let bitsPerComponent = 8
//            let colorSpace = CGColorSpaceCreateDeviceGray()
//            let context = CGContext(data: &data2, width: whiteGreenCleanTexture!.width, height: whiteGreenCleanTexture!.height, bitsPerComponent: bitsPerComponent, bytesPerRow: whiteGreenCleanTexture!.width, space: colorSpace, bitmapInfo: bitmapInfo.rawValue)
//
//
//            guard let dstImage = context?.makeImage() else { print("FAILED TO MAKE IMG")
//                return
//            }
//
//
//            let asUI = UIImage(cgImage: dstImage, scale: 0.0, orientation: .up)
//
//
//            UIGraphicsBeginImageContextWithOptions(asUI.size, true, 0)
//            let cgContext = UIGraphicsGetCurrentContext()!
//
//            asUI.draw(in: CGRect(origin: CGPoint(x: 0, y: 0), size: asUI.size))
//            UIColor.red.set()
//
//            var ellipseRect = CGRect(x: pt1[0] - 6, y: pt1[1] - 6, width: 12, height: 12)
//            cgContext.fillEllipse(in: ellipseRect)
//            
//            ellipseRect = CGRect(x: pt2[0] - 6, y: pt2[1] - 6, width: 12, height: 12)
//            cgContext.fillEllipse(in: ellipseRect)
//
//            ellipseRect = CGRect(x: pt3[0] - 6, y: pt3[1] - 6, width: 12, height: 12)
//            cgContext.fillEllipse(in: ellipseRect)
//
//            ellipseRect = CGRect(x: pt4[0] - 6, y: pt4[1] - 6, width: 12, height: 12)
//            cgContext.fillEllipse(in: ellipseRect)
//
//            ellipseRect = CGRect(x: pt5[0] - 6, y: pt5[1] - 6, width: 12, height: 12)
//            cgContext.fillEllipse(in: ellipseRect)
//
//            guard let withPoints = UIGraphicsGetImageFromCurrentImageContext() else {
//                return
//            }
            
            if (m >= 1) {
                print("JAEMS")
            }
        }
    }
    
    func fieldLogoDetection(width: Int, height: Int) -> [Float] {
        if (topBottomTexture != nil && whiteGreenCleanTexture != nil) {
            let tex = topBottomTexture!
            let bytesPerPixel = 2

            let bytesPerRow = tex.width * bytesPerPixel
            
            var data = [UInt16](repeating: 0, count: tex.width*tex.height)
            
            tex.getBytes(&data,
                bytesPerRow: bytesPerRow,
                from: MTLRegionMake2D(0, 0, tex.width, tex.height),
                mipmapLevel: 0)
            
            var run_length = 0
            var missed_sequential = 0
            var max_run_length = 0
            var max_run_index = -1
            var run_index = 0
            print(tex.width)
            print(tex.height)
            
//             Finds longest run of values
            for x in 0...(tex.width - 1) {
                let curr_min = Int(Double(data[x]) / Double(UInt16.max) * Double(height))
                let curr_max = Int(Double(data[x + tex.width]) / Double(UInt16.max) * Double(height))
                if (min(curr_min, curr_max) > 0) {
                    missed_sequential = 0
                    if (run_length == 0) {
                        run_index = x
                    }
                    run_length += 1
                } else {
                    if (missed_sequential) < 3 {
                        missed_sequential += 1
                    } else {
                        if (run_length > max_run_length) {
                            max_run_index = run_index
                            max_run_length = run_length
                        }
                        missed_sequential = 0
                        run_length = 0
                    }
                }
            }

            if (max_run_index < 0) {
                return []
            }

            var avg_y = 0
            
            var minsX: [Double] = []
            var minsY: [Double] = []

            // Finds center of pixel mass of the Cal logo
            for x in max_run_index..<(max_run_index + max_run_length) {
                let curr_min = Int(Double(data[x]) / Double(UInt16.max) * Double(height))
                let curr_max = Int(Double(data[x + tex.width]) / Double(UInt16.max) * Double(height))
                avg_y += (curr_max + curr_min) / 2
                minsY.append(Double(curr_max / 2 + curr_min / 2))
                minsX.append(Double(x))
            }
            avg_y /= max_run_length
            let avg_x = max_run_index + max_run_length / 2
            
            if (self.logoStart == nil) {
                self.logoStart = [avg_x, avg_y]
            } else {
                self.logoCoM = [avg_x, avg_y]
            }
            
            self.lastThree.append([avg_x, avg_y])
            if (self.lastThree.count > 6) {
                self.lastThree.remove(at: 0)
            }
            
            // Plots a triangle of the center of mass in Metal
            self.baryVertices = []
            let x1 = xImageSpacetoNormalized(x: avg_x - 8); let y1 = yImageSpacetoNormalized(y: avg_y + 8)
            self.baryVertices.append(Vertex(position: SIMD3(x1, y1, 0), color: SIMD4(0, 0, 1, 1)))
            let x2 = xImageSpacetoNormalized(x: avg_x + 8); let y2 = yImageSpacetoNormalized(y: avg_y + 8)
            self.baryVertices.append(Vertex(position: SIMD3(x2, y2, 0), color: SIMD4(0, 0, 1, 1)))
            let x3 = xImageSpacetoNormalized(x: avg_x); let y3 = yImageSpacetoNormalized(y: avg_y - 8)
            self.baryVertices.append(Vertex(position: SIMD3(x3, y3, 0), color: SIMD4(0, 0, 1, 1)))
            
            // Finds top line of the Cal Logo
            let bottomLogoLine = linearRegression(minsX, minsY)
            var slope = bottomLogoLine(1.0) - bottomLogoLine(0.0)
            if (abs(slope) < 0.001) {
                slope = 0.001
            }
            let perpendicularSlope = abs(0.0 - 1.0 / slope)
            let normalVec = normalize(SIMD2(1, perpendicularSlope))
            let theta = 0.045
            let rotated = SIMD2(normalVec.x * cos(theta) - normalVec.y * sin(theta), normalVec.x * sin(theta) + normalVec.y * cos(theta))
            
            let linex1 = xImageSpacetoNormalized(x: max_run_index); let liney1 = yImageSpacetoNormalized(y: Int(bottomLogoLine(Double(max_run_index))))
            let linex3 = xImageSpacetoNormalized(x: max_run_index  + max_run_length / 2); let liney3 = yImageSpacetoNormalized(y: 1)
            let linex2 = xImageSpacetoNormalized(x: max_run_index + max_run_length); let liney2 = yImageSpacetoNormalized(y: Int(bottomLogoLine(Double(max_run_index + max_run_length))))
            
            let endpointOneY = Int(bottomLogoLine(Double(max_run_index)))
            let endpointTwoY = Int(bottomLogoLine(Double(max_run_index + max_run_length)))
            
            
            self.baryVertices.append(Vertex(position: SIMD3(linex1, liney1, 0), color: SIMD4(1, 0, 1, 1)))
            self.baryVertices.append(Vertex(position: SIMD3(linex2, liney2, 0), color: SIMD4(1, 0, 1, 1)))
            self.baryVertices.append(Vertex(position: SIMD3(linex3, liney3, 0), color: SIMD4(1, 0, 1, 1)))
            
            return [Float(avg_x), Float(avg_y), Float(max_run_index), Float(endpointOneY), Float(max_run_index + max_run_length), Float(endpointTwoY), Float(normalVec.x), Float(normalVec.y), Float(rotated.x), Float(rotated.y)]
            
            
            if (numFrames % 90 == 108) {
                var data2 = [UInt8](repeating: 0, count: calLogoTexture!.width*calLogoTexture!.height)
                self.whiteGreenCleanTexture!.getBytes(&data2, bytesPerRow: bytesPerRow / 2, from: MTLRegionMake2D(0, 0, calLogoTexture!.width, calLogoTexture!.height), mipmapLevel: 0)

                let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
                let bitsPerComponent = 8
                let colorSpace = CGColorSpaceCreateDeviceGray()
                let context = CGContext(data: &data2, width: calLogoTexture!.width, height: calLogoTexture!.height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow / 2, space: colorSpace, bitmapInfo: bitmapInfo.rawValue)

                guard let dstImage = context?.makeImage() else { print("FAILED TO MAKE IMG")
                    return []
                }


                let asUI = UIImage(cgImage: dstImage, scale: 0.0, orientation: .up)


                UIGraphicsBeginImageContextWithOptions(asUI.size, true, 0)
                let cgContext = UIGraphicsGetCurrentContext()!

                asUI.draw(in: CGRect(origin: CGPoint(x: 0, y: 0), size: asUI.size))
                UIColor.red.set()
                
                let theta = 0.045
                let rotated = SIMD2(normalVec.x * cos(theta) - normalVec.y * sin(theta), normalVec.x * sin(theta) + normalVec.y * cos(theta))

                for i in 0...90 {
                    var ellipseRect = CGRect(x: max_run_index + Int(Double(i * 20) * rotated.x), y: endpointOneY + Int(Double(i * 20) * rotated.y), width: 8, height: 8)
                    cgContext.fillEllipse(in: ellipseRect)

                    ellipseRect = CGRect(x: max_run_index + max_run_length + Int(Double(i * 20) * normalVec.x), y: endpointTwoY + Int(Double(i * 20) * normalVec.y), width: 8, height: 8)
                    cgContext.fillEllipse(in: ellipseRect)

                    ellipseRect = CGRect(x: avg_x + Int(Double(i * 20) * normalVec.x), y: avg_y + Int(Double(i * 20) * normalVec.y), width: 8, height: 8)
                    cgContext.fillEllipse(in: ellipseRect)
                }
                
                UIColor.blue.set()
                
                // From right endpoint
                var i = 0; var numIntersections = 0; var sincePrev = 1000
                while (numIntersections < 2 && i < 400) {
                    i += 1
                    let x = max_run_index + max_run_length + Int(Double(i) * normalVec.x)
                    let y = endpointTwoY + Int(Double(i) * normalVec.y)
                    let texCoord = x + y * whiteGreenCleanTexture.width
                    if (pixelNearby(x: x, y: y, data2: data2) && sincePrev > 30) {
//                        let ellipseRect = CGRect(x: x, y: y, width: 8, height: 8)
//                        cgContext.fillEllipse(in: ellipseRect)
                        addMetalTriangleVerts(x: x, y: y)
                        
//                        let x1 = xImageSpacetoNormalized(x: x); let y1 = yImageSpacetoNormalized(y: y + 8)
//                        self.baryVertices.append(Vertex(position: SIMD3(x1, y1, 0), color: SIMD4(0, 0, 1, 1)))
//                        let x2 = xImageSpacetoNormalized(x: x + 8); let y2 = yImageSpacetoNormalized(y: y + 8)
//                        self.baryVertices.append(Vertex(position: SIMD3(x2, y2, 0), color: SIMD4(0, 0, 1, 1)))
//                        let x3 = xImageSpacetoNormalized(x: x); let y3 = yImageSpacetoNormalized(y: y - 8)
//                        self.baryVertices.append(Vertex(position: SIMD3(x3, y3, 0), color: SIMD4(0, 0, 1, 1)))
                        sincePrev = 0
                        numIntersections += 1
                    } else {
                        sincePrev += 1
                    }
                    
                }
                
                
                
                // From left endpoint
                i = 0; numIntersections = 0; sincePrev = 1000
                while (numIntersections < 1 && i < 400) {
                    i += 1
                    let x = max_run_index + Int(Double(i) * rotated.x)
                    let y = endpointOneY + Int(Double(i) * rotated.y)
                    let texCoord = x + y * whiteGreenCleanTexture.width
                    if (pixelNearby(x: x, y: y, data2: data2) && sincePrev > 30) {
                        addMetalTriangleVerts(x: x, y: y)
                        sincePrev = 0
                        numIntersections += 1
                    } else {
                        sincePrev += 1
                    }
                    
                }
                
                
                
                // From center of mass
                i = 0; numIntersections = 0; sincePrev = 1000
                while (numIntersections < 2 && i < 400) {
                    i += 1
                    let x = avg_x + Int(Double(i) * normalVec.x)
                    let y = avg_y + Int(Double(i) * normalVec.y)
                    let texCoord = x + y * whiteGreenCleanTexture.width
                    if (pixelNearby(x: x, y: y, data2: data2) && sincePrev > 30) {
//                        let ellipseRect = CGRect(x: x, y: y, width: 8, height: 8)
//                        cgContext.fillEllipse(in: ellipseRect)
                        addMetalTriangleVerts(x: x, y: y)
                        sincePrev = 0
                        numIntersections += 1
                    } else {
                        sincePrev += 1
                    }
                    
                }
                


//                UIColor.yellow.set()
//                var ellipseRect = CGRect(x: avg_x - 7, y: avg_y - 7, width: 14, height: 14)
//                cgContext.fillEllipse(in: ellipseRect)
//
//
//
//                guard let withPoints = UIGraphicsGetImageFromCurrentImageContext() else {
//                    return
//                }
                
                
                print("hello")
                
            }
        }
        return []
    }
    
    func addMetalTriangleVerts(x: Int, y: Int) {
        var i = xImageSpacetoNormalized(x: x + 9); var j = yImageSpacetoNormalized(y: y + 9)
        self.baryVertices.append(Vertex(position: SIMD3(i, j, 0), color: SIMD4(1, 0, 0, 1)))
        i = xImageSpacetoNormalized(x: x - 9); j = yImageSpacetoNormalized(y: y + 9)
        self.baryVertices.append(Vertex(position: SIMD3(i, j, 0), color: SIMD4(1, 0, 0, 1)))
        i = xImageSpacetoNormalized(x: x); j = yImageSpacetoNormalized(y: y - 9)
        self.baryVertices.append(Vertex(position: SIMD3(i, j, 0), color: SIMD4(1, 0, 0, 1)))
    }
    
    func pixelNearby(x: Int, y: Int, data2: UnsafePointer<UInt8>) -> Bool{
        for i in x-2...x+2 {
            for j in y-2...y+2 {
                let texCoord = i + j * whiteGreenCleanTexture.width
                if (data2[texCoord] > 10) {
                    return true
                }
            }
        }
        return false
    }
    
//    let firstMinX = 2.0 * (Float(firstBoxMinMedian.0) / Float(width)) - 1.0
//    let firstMinY = 0.0 - (2.0 * (Float(firstBoxMinMedian.1) / Float(height)) - 1.0)
    
    func xImageSpacetoNormalized(x: Int) -> Float {
        return 2.0 * (Float(x) / Float(self.tileOutTexture.width)) - 1.0
    }
    
    func yImageSpacetoNormalized(y: Int) -> Float {
        return 0.0 - (2.0 * (Float(y) / Float(self.tileOutTexture.height)) - 1.0)
    }
    
    func logoPointDetection(width: Int, height: Int) {
        
        if (topBottomTexture != nil) {
            let tex = topBottomTexture!
            let bytesPerPixel = 2

            let bytesPerRow = tex.width * bytesPerPixel
            
            var data = [UInt16](repeating: 0, count: tex.width*tex.height)
            
            tex.getBytes(&data,
                bytesPerRow: bytesPerRow,
                from: MTLRegionMake2D(0, 0, tex.width, tex.height),
                mipmapLevel: 0)
            

            var run_length = 0
            var zeroes = data[0] < 1
            var runs: [[Int]] = []
            var prevEnd = 0
            
            let t = tex.width
            for x in 0...(tex.width - 1) {
                let curr_min = Int(Double(data[x]) / Double(UInt16.max) * Double(height))
                let curr_max = Int(Double(data[x + tex.width]) / Double(UInt16.max) * Double(height))
                let zero = max(curr_max, curr_min) == 0
                if (zero != zeroes && run_length > 4) {
                    zeroes = !zeroes
                    runs.append([prevEnd, run_length + prevEnd])
                    prevEnd = prevEnd + run_length + 1
                    run_length = 0
                } else {
                    run_length += 1
                }

            }
            
            runs.append([prevEnd, run_length + prevEnd])
            
            self.baryVertices = []
            
            let firstBoxYMins = Array(data[runs[1][0]-1..<runs[1][1]])
            let firstBoxXs = Array(runs[1][0]-1..<runs[1][1])
            var firstBoxMins = zip(firstBoxXs, firstBoxYMins).sorted(by: {UInt16($0.0) > UInt16($1.0)})
            var firstBoxMinMedian = firstBoxMins[firstBoxMins.count / 2]
            firstBoxMins.removeAll(where: {abs(Int($1) - Int(firstBoxMinMedian.1)) > 100})
            var firstBoxXAvg = 0
            var firstBoxYAvg: UInt16 = 0
            for boxMin in firstBoxMins {
                firstBoxXAvg += boxMin.0 / firstBoxMins.count
                firstBoxYAvg += boxMin.1 / UInt16(firstBoxMins.count)
            }
            
//            firstBoxMinMedian = firstBoxMins[0]
//            firstBoxMinMedian.1 = UInt16(Int(Double(firstBoxMinMedian.1) / Double(UInt16.max) * Double(height)))
            firstBoxMinMedian.0 = firstBoxXAvg
            firstBoxMinMedian.1 = UInt16(Int(Double(firstBoxMinMedian.1) / Double(UInt16.max) * Double(height)))
            let firstMinX = 2.0 * (Float(firstBoxMinMedian.0) / Float(width)) - 1.0
            let firstMinY = 0.0 - (2.0 * (Float(firstBoxMinMedian.1) / Float(height)) - 1.0)
            
            self.baryVertices.append(Vertex(position: SIMD3(firstMinX, firstMinY, 0), color: SIMD4(0.91, 0.812, 0.33, 1)))
            
            // 232, 207, 84
            


            let firstBoxYMaxes = Array(data[tex.width + runs[1][0]-1..<tex.width + runs[1][1]])
            var firstBoxMaxes = zip(firstBoxXs, firstBoxYMaxes).sorted(by: {UInt16($0.0) > UInt16($1.0)})
            var firstBoxMaxMedian = firstBoxMaxes[firstBoxMaxes.count / 2]
            firstBoxMaxes.removeAll(where: {abs(Int($1) - Int(firstBoxMaxMedian.1)) > 100})
            firstBoxMaxMedian = firstBoxMaxes[0]
            firstBoxMaxMedian.1 = UInt16(Int(Double(firstBoxMaxMedian.1) / Double(UInt16.max) * Double(height)))
//            self.baryVertices.append([firstBoxMaxMedian.0, Int(firstBoxMaxMedian.1)])
//            self.baryVertices.append(2.0 * (Float(firstBoxMaxMedian.0) / Float(width)) - 1.0)
//            self.baryVertices.append(2.0 * (Float(firstBoxMaxMedian.1) / Float(height)) - 1.0)
            let firstMaxX = 2.0 * (Float(firstBoxMaxMedian.0) / Float(width)) - 1.0
            let firstMaxY = 0.0 - (2.0 * (Float(firstBoxMaxMedian.1) / Float(height)) - 1.0)
            
            
            // 248, 117, 250
            
            self.baryVertices.append(Vertex(position: SIMD3(firstMaxX, firstMaxY, 0), color: SIMD4(0.97, 0.46, 0.98, 1)))

            let secondBoxYMins = Array(data[runs[5][0]-1..<runs[5][1]])
            let secondBoxXs = Array(runs[5][0]-1..<runs[5][1])
            var secondBoxMins = zip(secondBoxXs, secondBoxYMins).sorted(by: {UInt16($0.0) < UInt16($1.0)})
            var secondBoxMinMedian = secondBoxMins[secondBoxMins.count / 2]
            secondBoxMins.removeAll(where: {abs(Int($1) - Int(secondBoxMinMedian.1)) > 100})
            secondBoxMinMedian = secondBoxMins[0]
            secondBoxMinMedian.1 = UInt16(Int(Double(secondBoxMinMedian.1) / Double(UInt16.max) * Double(height)))
            
            let secondMinX = 2.0 * (Float(secondBoxMinMedian.0) / Float(width)) - 1.0
            let secondMinY = 0.0 - (2.0 * (Float(secondBoxMinMedian.1) / Float(height)) - 1.0)
            self.baryVertices.append(Vertex(position: SIMD3(secondMinX, secondMinY, 0), color: SIMD4(0.61, 0.97, 0.51, 1)))
            
            // 255, 117, 138

            let secondBoxYMaxes = Array(data[tex.width + runs[5][0]-1..<tex.width + runs[5][1]])
            var secondBoxMaxes = zip(secondBoxXs, secondBoxYMaxes).sorted(by: {UInt16($0.0) < UInt16($1.0)})
            var secondBoxMaxMedian = secondBoxMaxes[secondBoxMaxes.count / 2]
            secondBoxMaxes.removeAll(where: {abs(Int($1) - Int(secondBoxMaxMedian.1)) > 100})
            secondBoxMaxMedian = secondBoxMaxes[0]
            secondBoxMaxMedian.1 = UInt16(Int(Double(secondBoxMaxMedian.1) / Double(UInt16.max) * Double(height)))
            
            let secondMaxX = 2.0 * (Float(secondBoxMaxMedian.0) / Float(width)) - 1.0
            let secondMaxY = 0.0 - (2.0 * (Float(secondBoxMaxMedian.1) / Float(height)) - 1.0)
            
            // 117, 250, 248
            self.baryVertices.append(Vertex(position: SIMD3(secondMaxX, secondMaxY, 0), color: SIMD4(0.46, 0.98, 0.972, 1)))
            
            
            
            if (numFrames % 30 == 0) {
                var data2 = [UInt8](repeating: 0, count: calLogoTexture!.width*calLogoTexture!.height)
                self.calLogoTexture!.getBytes(&data2, bytesPerRow: bytesPerRow / 2, from: MTLRegionMake2D(0, 0, calLogoTexture!.width, calLogoTexture!.height), mipmapLevel: 0)

                let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
                let bitsPerComponent = 8
                let colorSpace = CGColorSpaceCreateDeviceGray()
                let context = CGContext(data: &data2, width: calLogoTexture!.width, height: calLogoTexture!.height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow / 2, space: colorSpace, bitmapInfo: bitmapInfo.rawValue)

                guard let dstImage = context?.makeImage() else { print("FAILED TO MAKE IMG")
                    return
                }


                let asUI = UIImage(cgImage: dstImage, scale: 0.0, orientation: .up)
                
                
                UIGraphicsBeginImageContextWithOptions(asUI.size, true, 0)
                let cgContext = UIGraphicsGetCurrentContext()!

                asUI.draw(in: CGRect(origin: CGPoint(x: 0, y: 0), size: asUI.size))
                UIColor.red.set()


                var ellipseRect = CGRect(x: firstBoxMinMedian.0 - 7, y: Int(firstBoxMinMedian.1) - 7, width: 14, height: 14)
                cgContext.fillEllipse(in: ellipseRect)

                ellipseRect = CGRect(x: firstBoxMaxMedian.0 - 7, y: Int(firstBoxMaxMedian.1) - 7, width: 14, height: 14)
                cgContext.fillEllipse(in: ellipseRect)

                ellipseRect = CGRect(x: secondBoxMinMedian.0 - 7, y: Int(secondBoxMinMedian.1) - 7, width: 14, height: 14)
                cgContext.fillEllipse(in: ellipseRect)

                ellipseRect = CGRect(x: secondBoxMaxMedian.0 - 7, y: Int(secondBoxMaxMedian.1) - 7, width: 14, height: 14)
                cgContext.fillEllipse(in: ellipseRect)


                guard let withPoints = UIGraphicsGetImageFromCurrentImageContext() else {
                    return
                }
                
                var tf = 2
                tf += 4
            }
            
            
        }
    }
    
    
    func findLines() async {
        
        
        if (filteredYTexture != nil) {
            
            let tex = filteredYTexture!
            
            let bytesPerPixel = 1
            
            let bytesPerRow = tex.width * bytesPerPixel
            
            var data = [UInt8](repeating: 0, count: tex.width*tex.height)
            
            tex.getBytes(&data,
                bytesPerRow: bytesPerRow,
                from: MTLRegionMake2D(0, 0, tex.width, tex.height),
                mipmapLevel: 0)
            
            
            let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
            let bitsPerComponent = 8
            let colorSpace = CGColorSpaceCreateDeviceGray()
            let context = CGContext(data: &data, width: tex.width, height: tex.height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo.rawValue)
            
            let m = data.max()!

            // Creates the image from the graphics context
            guard let dstImage = context?.makeImage() else { print("FAILED TO MAKE IMG")
                return
            }
            
            
            let asUI = UIImage(cgImage: dstImage, scale: 0.0, orientation: .up)
            
            print("TRANSFORMING")
            let hough = houghSpace(image: asUI, data: data)
            print("FINDING PEAKS")
            newLines = peakLines(houghSpace: hough, n: 10)
            let imageWithLines = draw(lines: newLines, inImage: asUI, color: .red)
            lineUpdate += 1
            frameProcessed = true
            if (m > 10) {
                print("JAEMS")
            }
            return
            
            
        }
    }
    
    private func buildBaryPipelineState(library: MTLLibrary) {
        let vertexFunction = library.makeFunction(name: "baryTransform")
        let fragmentFunction = library.makeFunction(name: "baryFragmentShader")
        
        let baryPipelineDescriptor = MTLRenderPipelineDescriptor()
        baryPipelineDescriptor.vertexFunction = vertexFunction
        baryPipelineDescriptor.fragmentFunction = fragmentFunction
        baryPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float3
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        
        vertexDescriptor.attributes[1].format = .float4
        vertexDescriptor.attributes[1].offset = MemoryLayout<float3>.stride
        vertexDescriptor.attributes[1].bufferIndex = 0
        
        vertexDescriptor.layouts[0].stride = MemoryLayout<Vertex>.stride
        
        baryPipelineDescriptor.vertexDescriptor = vertexDescriptor
        
        do {
            baryPipelineState = try device.makeRenderPipelineState(descriptor: baryPipelineDescriptor)
        } catch let error as NSError {
            print("error: \(error.localizedDescription)")
        }
    }
    
    
    // MARK: - Private
    
    // Create and load our basic Metal state objects.
    func loadMetal() {
        // Set the default formats needed to render.
        renderDestination.colorPixelFormat = .bgra8Unorm
        renderDestination.sampleCount = 1
        
        // Create a vertex buffer with our image plane vertex data.
        let imagePlaneVertexDataCount = kImagePlaneVertexData.count * MemoryLayout<Float>.size
        imagePlaneVertexBuffer = device.makeBuffer(bytes: kImagePlaneVertexData, length: imagePlaneVertexDataCount, options: [])
        imagePlaneVertexBuffer.label = "ImagePlaneVertexBuffer"
        
        // Load all the shader files with a metal file extension in the project.
        let defaultLibrary = device.makeDefaultLibrary()!
        
        buildBaryPipelineState(library: defaultLibrary)
        
                
        // Create a vertex descriptor for our image plane vertex buffer.
        let imagePlaneVertexDescriptor = MTLVertexDescriptor()
        
        // Positions.
        imagePlaneVertexDescriptor.attributes[0].format = .float2
        imagePlaneVertexDescriptor.attributes[0].offset = 0
        imagePlaneVertexDescriptor.attributes[0].bufferIndex = Int(kBufferIndexMeshPositions.rawValue)
        
        // Texture coordinates.
        imagePlaneVertexDescriptor.attributes[1].format = .float2
        imagePlaneVertexDescriptor.attributes[1].offset = 8
        imagePlaneVertexDescriptor.attributes[1].bufferIndex = Int(kBufferIndexMeshPositions.rawValue)
        
        // Buffer Layout.
        imagePlaneVertexDescriptor.layouts[0].stride = 16
        imagePlaneVertexDescriptor.layouts[0].stepRate = 1
        imagePlaneVertexDescriptor.layouts[0].stepFunction = .perVertex
                        
        // Create camera image texture cache.
        var textureCache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
        cameraImageTextureCache = textureCache
        
        // Define the shaders that will render the camera image and fog on the GPU.
        let fogVertexFunction = defaultLibrary.makeFunction(name: "fogVertexTransform")!
        let fogFragmentFunction = defaultLibrary.makeFunction(name: "fogFragmentShader")!
        let fogPipelineStateDescriptor = MTLRenderPipelineDescriptor()
        fogPipelineStateDescriptor.label = "MyFogPipeline"
        fogPipelineStateDescriptor.sampleCount = renderDestination.sampleCount
        fogPipelineStateDescriptor.vertexFunction = fogVertexFunction
        fogPipelineStateDescriptor.fragmentFunction = fogFragmentFunction
        
        
        
        fogPipelineStateDescriptor.vertexDescriptor = imagePlaneVertexDescriptor
        fogPipelineStateDescriptor.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat

        // Initialize the pipeline.
        do {
            try fogPipelineState = device.makeRenderPipelineState(descriptor: fogPipelineStateDescriptor)
        } catch let error {
            print("Failed to create fog pipeline state, error \(error)")
        }
        
//        var computePipeline : MTLComputePipelineState;
//        computePipeline.label = "MyComputePipeline"
        let f : MTLFunction = defaultLibrary.makeFunction(name: "colorTransform")!
        do {
            self.computePipelineState = try device.makeComputePipelineState(function: f)
        } catch let error {
            print("Failed to create compute pipeline state, error \(error)")
        }
        
        let f2 : MTLFunction = defaultLibrary.makeFunction(name: "imageClean")!
        do {
            self.cleanComputePipelineState = try device.makeComputePipelineState(function: f2)
        } catch let error {
            print("Failed to create second compute pipeline state, error \(error)")
        }
        
        do {
            self.greenWhiteCleanComputePipelineState = try device.makeComputePipelineState(function: f2)
        } catch let error {
            print("Failed to create second compute pipeline state, error \(error)")
        }
        
        let f3 : MTLFunction = defaultLibrary.makeFunction(name: "logoDetect")!
        do {
            self.logoComputePipelineState = try device.makeComputePipelineState(function: f3)
        } catch let error {
            print("Failed to create second compute pipeline state, error \(error)")
        }
        
        let f4 : MTLFunction = defaultLibrary.makeFunction(name: "raycastFromLogo")!
        do {
            self.logoRaycastComputePipelineState = try device.makeComputePipelineState(function: f4)
        } catch let error {
            print("Failed to create second compute pipeline state, error \(error)")
        }
//        do {
//        try tilePipelineState = device.makeRenderPipelineState(tileDescriptor: tileDescriptor, options: tileOption, reflection: nil)
//        } catch let error {
//            print("Failed to create tile pipeline state, error \(error)")
//        }
        
        
        
        
        // Create the command queue for one frame of rendering work.
        commandQueue = device.makeCommandQueue()
    }
    
    // Updates any app state.
    func updateAppState() {

        // Get the AR session's current frame.
        guard let currentFrame = session.currentFrame else {
            return
        }
        
        // Prepare the current frame's camera image for transfer to the GPU.
        updateCameraImageTextures(frame: currentFrame)
        
        // Prepare the current frame's depth and confidence images for transfer to the GPU.
        updateARDepthTexures(frame: currentFrame)
        
        // Update the destination-rendering vertex info if the size of the screen changed.
        if viewportSizeDidChange {
            viewportSizeDidChange = false
            updateImagePlane(frame: currentFrame)
        }
    }
        
    // Creates two textures (Y and CbCr) to transfer the current frame's camera image to the GPU for rendering.
    func updateCameraImageTextures(frame: ARFrame) {
        if CVPixelBufferGetPlaneCount(frame.capturedImage) < 2 {
            return
        }
        cameraImageTextureY = createTexture(fromPixelBuffer: frame.capturedImage, pixelFormat: .r8Unorm, planeIndex: 0)
        cameraImageTextureCbCr = createTexture(fromPixelBuffer: frame.capturedImage, pixelFormat: .rg8Unorm, planeIndex: 1)
        
//        print(frame.camera.eulerAngles)
        
    }

    // Assigns an appropriate MTL pixel format given the argument pixel-buffer's format.
    fileprivate func setMTLPixelFormat(_ texturePixelFormat: inout MTLPixelFormat?, basedOn pixelBuffer: CVPixelBuffer!) {
        if CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_DepthFloat32 {
            texturePixelFormat = .r32Float
        } else if CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_OneComponent8 {
            texturePixelFormat = .r8Uint
        } else {
            fatalError("Unsupported ARDepthData pixel-buffer format.")
        }
    }

    // Prepares the scene depth information for transfer to the GPU for rendering.
    func updateARDepthTexures(frame: ARFrame) {
        // Get the scene depth or smoothed scene depth from the current frame.
        guard let sceneDepth = frame.smoothedSceneDepth ?? frame.sceneDepth else {
            print("Failed to acquire scene depth.")
            return
        }
        var pixelBuffer: CVPixelBuffer!
        pixelBuffer = sceneDepth.depthMap
        
        // Set up the destination pixel format for the depth information, and
        // create a Metal texture from the depth image provided by ARKit.
        var texturePixelFormat: MTLPixelFormat!
        setMTLPixelFormat(&texturePixelFormat, basedOn: pixelBuffer)
        depthTexture = createTexture(fromPixelBuffer: pixelBuffer, pixelFormat: texturePixelFormat, planeIndex: 0)

        // Get the current depth confidence values from the current frame.
        // Set up the destination pixel format for the confidence information, and
        // create a Metal texture from the confidence image provided by ARKit.
        pixelBuffer = sceneDepth.confidenceMap
        setMTLPixelFormat(&texturePixelFormat, basedOn: pixelBuffer)
        confidenceTexture = createTexture(fromPixelBuffer: pixelBuffer, pixelFormat: texturePixelFormat, planeIndex: 0)
        
    }
        
    // Creates a Metal texture with the argument pixel format from a CVPixelBuffer at the argument plane index.
    func createTexture(fromPixelBuffer pixelBuffer: CVPixelBuffer, pixelFormat: MTLPixelFormat, planeIndex: Int) -> CVMetalTexture? {
        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, planeIndex)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, planeIndex)
        
        var texture: CVMetalTexture? = nil
        let status = CVMetalTextureCacheCreateTextureFromImage(nil, cameraImageTextureCache, pixelBuffer, nil, pixelFormat,
                                                               width, height, planeIndex, &texture)
        
        if status != kCVReturnSuccess {
            texture = nil
        }
        
        return texture
    }
    
    // Sets up vertex data (source and destination rectangles) rendering.
    func updateImagePlane(frame: ARFrame) {
        // Update the texture coordinates of the image plane to aspect fill the viewport.
        let displayToCameraTransform = frame.displayTransform(for: .landscapeRight, viewportSize: viewportSize).inverted()
        let vertexData = imagePlaneVertexBuffer.contents().assumingMemoryBound(to: Float.self)
        let fogVertexData = imagePlaneVertexBuffer.contents().assumingMemoryBound(to: Float.self)
        for index in 0...3 {
            let textureCoordIndex = 4 * index + 2
            let textureCoord = CGPoint(x: CGFloat(kImagePlaneVertexData[textureCoordIndex]), y: CGFloat(kImagePlaneVertexData[textureCoordIndex + 1]))
            let transformedCoord = textureCoord.applying(displayToCameraTransform)
            vertexData[textureCoordIndex] = Float(transformedCoord.x)
            vertexData[textureCoordIndex + 1] = Float(transformedCoord.y)
            fogVertexData[textureCoordIndex] = Float(transformedCoord.x)
            fogVertexData[textureCoordIndex + 1] = Float(transformedCoord.y)
        }
    }
    
    func doBaryPass(renderEncoder: MTLRenderCommandEncoder) {
        if (self.baryVertices.count > 0) {
            let vertices: [Float] = [0, 1, 0,
                                     -1, -1, 0,
                                     1, -1, 0]
            
//            var indices: [UInt16] = []
//
//            let numTris = 7
//            for i in 1...numTris * 3 {
//                indices.append(UInt16(i))
//            }
            
            let indices: [UInt16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            
            let vertexBuffer = device.makeBuffer(bytes: self.baryVertices, length: self.baryVertices.count * MemoryLayout<Vertex>.size, options: [])
            
            let indexBuffer = device.makeBuffer(bytes: indices, length: indices.count * MemoryLayout<UInt16>.size, options: [])
            
            renderEncoder.setRenderPipelineState(self.baryPipelineState)
            renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            renderEncoder.drawIndexedPrimitives(type: .triangle, indexCount: indices.count, indexType: .uint16, indexBuffer: indexBuffer!, indexBufferOffset: 0)
        }
    }
    
    // Schedules the camera image and fog to be rendered on the GPU.
    func doFogRenderPass(renderEncoder: MTLRenderCommandEncoder) {
        guard let cameraImageY = cameraImageTextureY, let cameraImageCbCr = cameraImageTextureCbCr,
            let confidenceTexture = confidenceTexture else {
            return
        }

        // Push a debug group that enables you to identify this render pass in a Metal frame capture.
        renderEncoder.pushDebugGroup("FogPass")

        // Set render command encoder state.
        renderEncoder.setCullMode(.none)
        renderEncoder.setRenderPipelineState(fogPipelineState)

        // Setup plane vertex buffers.
        renderEncoder.setVertexBuffer(imagePlaneVertexBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(imagePlaneVertexBuffer, offset: 0, index: 1)

        // Setup textures for the fog fragment shader.
//        renderEncoder.setFragmentTexture(cleanOutTexture, index: 0)
//        renderEncoder.setFragmentTexture(tileOutTexture, index: 0)
//        renderEncoder.setFragmentTexture(calLogoTexture, index: 0)
//        renderEncoder.setFragmentTexture(filteredYTexture, index: 0)
        renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(cameraImageY), index: 0)
        renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(cameraImageCbCr), index: 1)
        renderEncoder.setFragmentTexture(filteredDepthTexture, index: 2)
        renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(confidenceTexture), index: 3)
        renderEncoder.setFragmentTexture(sceneTexture, index: 4)
        // Draw final quad to display
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        
        
        renderEncoder.popDebugGroup()
        
        
    }
    
    // MARK: - MPS Filter
    
    // Sets up a filter to process the depth texture.
    func setupFilter(width: Int, height: Int) {
        // Create a destination backing-store to hold the blurred result.
        let filteredDepthDescriptor = MTLTextureDescriptor()
        filteredDepthDescriptor.pixelFormat = .r32Float
        filteredDepthDescriptor.width = width
        filteredDepthDescriptor.height = height
        filteredDepthDescriptor.usage = [.shaderRead, .shaderWrite]
//        filteredYTexture = device.makeTexture(descriptor: filteredDepthDescriptor)
        filteredDepthTexture = device.makeTexture(descriptor: filteredDepthDescriptor)
        downsizeFilter = MPSImageLanczosScale(device: device)
    }
    
    // Sets up a filter to process the depth texture.
    func setupYFilter(width: Int, height: Int) {
        // Create a destination backing-store to hold the blurred result.
        let filteredYDescriptor = MTLTextureDescriptor()
        filteredYDescriptor.pixelFormat = .r8Unorm
        filteredYDescriptor.width = width
        filteredYDescriptor.height = height
        filteredYDescriptor.usage = [.shaderRead, .shaderWrite]
        filteredYTexture = device.makeTexture(descriptor: filteredYDescriptor)
        
        let tileOutDescriptor = MTLTextureDescriptor()
        tileOutDescriptor.pixelFormat = .r8Unorm
        tileOutDescriptor.width = width
        tileOutDescriptor.height = height
        tileOutDescriptor.usage = [.shaderRead, .shaderWrite]
        tileOutTexture = device.makeTexture(descriptor: tileOutDescriptor)
        
        let cleanOutDescriptor = MTLTextureDescriptor()
        cleanOutDescriptor.pixelFormat = .r8Unorm
        cleanOutDescriptor.width = width
        cleanOutDescriptor.height = height
        cleanOutDescriptor.usage = [.shaderRead, .shaderWrite]
        cleanOutTexture = device.makeTexture(descriptor: cleanOutDescriptor)
        
//        let cleanOutDescriptor = MTLTextureDescriptor()
//        cleanOutDescriptor.pixelFormat = .r8Unorm
//        cleanOutDescriptor.width = width
//        cleanOutDescriptor.height = height
//        cleanOutDescriptor.usage = [.shaderRead, .shaderWrite]
        whiteGreenCleanTexture = device.makeTexture(descriptor: cleanOutDescriptor)
        
        let calLogoDescriptor = MTLTextureDescriptor()
        calLogoDescriptor.pixelFormat = .r8Unorm
        calLogoDescriptor.width = width
        calLogoDescriptor.height = height
        calLogoDescriptor.usage = [.shaderRead, .shaderWrite]
        calLogoTexture = device.makeTexture(descriptor: calLogoDescriptor)
        
        let topBottomDescriptor = MTLTextureDescriptor()
        topBottomDescriptor.pixelFormat = .r16Unorm
        topBottomDescriptor.width = width
        topBottomDescriptor.height = 2
        topBottomDescriptor.usage = [.shaderRead, .shaderWrite]
        topBottomTexture = device.makeTexture(descriptor: topBottomDescriptor)
        
        let logoRaycastDescriptor = MTLTextureDescriptor()
        logoRaycastDescriptor.pixelFormat = .r16Unorm
        logoRaycastDescriptor.width = 10
        logoRaycastDescriptor.height = 1
        logoRaycastDescriptor.usage = [.shaderRead, .shaderWrite]
        logoRaycastTexture = device.makeTexture(descriptor: logoRaycastDescriptor)
        
        let logoRaycastOutDescriptor = MTLTextureDescriptor()
        logoRaycastOutDescriptor.pixelFormat = .r16Unorm
        logoRaycastOutDescriptor.width = 10
        logoRaycastOutDescriptor.height = 1
        logoRaycastOutDescriptor.usage = [.shaderRead, .shaderWrite]
        logoRaycastOutTexture = device.makeTexture(descriptor: logoRaycastOutDescriptor)
        
        let sceneOutDescriptor = MTLTextureDescriptor()
        sceneOutDescriptor.pixelFormat = .rgba8Unorm
        sceneOutDescriptor.width = width
        sceneOutDescriptor.height = height
//        sceneOutDescriptor.storageMode = .private
        sceneOutDescriptor.usage = MTLTextureUsage(rawValue: MTLTextureUsage.renderTarget.rawValue | MTLTextureUsage.shaderRead.rawValue)
        sceneTexture = device.makeTexture(descriptor: sceneOutDescriptor)
        
//        blurFilter = MPSImageGaussianBlur(device: device, sigma: 8)
        let luminanceWeights: [Float] = [ 0.333, 0.334, 0.333 ]
        sobelFilter = MPSImageSobel(device: device)
    }
    
    // Sets up a filter to process the depth texture.
    func setupLines(width: Int, height: Int) {
        // Create a destination backing-store to hold the blurred result.
        let lineDescriptor = MTLTextureDescriptor()
        lineDescriptor.pixelFormat = .r8Unorm
        lineDescriptor.width = width
        lineDescriptor.height = height
        lineDescriptor.usage = [.shaderRead, .shaderWrite]
        lineTexture = device.makeTexture(descriptor: lineDescriptor)
    }
    
    // Schedules the depth texture to be blurred on the GPU using the `blurFilter`.
    func applyGaussianBlur(commandBuffer: MTLCommandBuffer) {
        guard let depthTexture = filteredYTexture else {
            print("Error: Unable to apply the MPS filter.")
            return
        }
        guard let lancoz = downsizeFilter else {
            setupFilter(width: depthTexture.width, height: depthTexture.height)
            return
        }
        
        let inputImage = MPSImage(texture: depthTexture, featureChannels: 1)
        let outputImage = MPSImage(texture: CVMetalTextureGetTexture(cameraImageTextureY! )!, featureChannels: 1)
        lancoz.encode(commandBuffer: commandBuffer, sourceImage: inputImage, destinationImage: outputImage)
    }
    
    
    // Schedules the depth texture to be blurred on the GPU using the `blurFilter`.
    func applySobel(commandBuffer: MTLCommandBuffer) {
        guard let yTexture = tileOutTexture else {
            print("Error: Unable to apply the MPS filter.")
            return
        }
        let sobel = sobelFilter!
        
        
        let inputImage = MPSImage(texture: yTexture, featureChannels: 1)
        let outputImage = MPSImage(texture: filteredYTexture, featureChannels: 1)
        sobel.encode(commandBuffer: commandBuffer, sourceImage: inputImage, destinationImage: outputImage)
    }
}
