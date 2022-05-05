import UIKit
import CoreGraphics

let CLAMP_THRESHOLD = 225.0

/*
     Matrix
*/

typealias Vector = [Double]
typealias Matrix = [Vector]

fileprivate func size(ofMatrix matrix: Matrix) -> CGSize {
    if matrix.count == 0 { return CGSize.zero }
    
    let firstRow = matrix[0]
    return CGSize(width: firstRow.count, height: matrix.count)
}

fileprivate func emptyMatrix(size: CGSize) -> Matrix {
    return Matrix(
        repeating: Vector(repeating: 0, count: Int(size.width)),
        count: Int(size.height)
    )
}

fileprivate func vectorize(matrix: Matrix) -> Vector {
    let matrixSize = size(ofMatrix: matrix)
    var vector = Vector(
        repeating: 0,
        count: Int(matrixSize.width * matrixSize.height)
    )
    
    for (i, row) in matrix.enumerated() {
        for (j, _) in row.enumerated() {
            vector[i * Int(matrixSize.width) + j] = matrix[i][j]
        }
    }
    
    return vector
}

/*
     Image helpers
*/

typealias Line = (
    theta: Int,
    distance: Int,
    occurrence: Int
)

fileprivate func createImageContext(data: UnsafeMutableRawPointer, size: CGSize)
        -> CGContext {
    let width = Int(size.width)
    let height = Int(size.height)
    
    let colorspace = CGColorSpaceCreateDeviceGray()
    let context = CGContext(data: data, width: width, height: height,
                            bitsPerComponent: 8, bytesPerRow: width,
                            space: colorspace,
                            bitmapInfo: CGImageAlphaInfo.none.rawValue)
    
    return context!
}

fileprivate func makeImage(matrix: Matrix) -> UIImage {
    var vectorized = vectorize(matrix: matrix).map { UInt8($0) }
    let context = createImageContext(data: &vectorized,
                                     size: size(ofMatrix: matrix))
    
    let image = context.makeImage()
    return UIImage(cgImage: image!)
}

fileprivate func transformForRendering(houghSpace matrix: Matrix) -> Matrix {
    var maxValue = 0.0
    for (i, row) in matrix.enumerated() {
        for (j, _) in row.enumerated() {
            maxValue = max(matrix[i][j], maxValue)
        }
    }
    
    var transformedMatrix = matrix
    for (i, row) in matrix.enumerated() {
        for (j, _) in row.enumerated() {
            transformedMatrix[i][j] = matrix[i][j] * 255.0 / maxValue
        }
    }
    
    return transformedMatrix
}

func draw(lines: [Line], inImage image: UIImage, color: UIColor) -> UIImage {
    UIGraphicsBeginImageContextWithOptions(image.size, true, 0)
    let context = UIGraphicsGetCurrentContext()!
    
    image.draw(in: CGRect(origin: CGPoint(x: 0, y: 0), size: image.size))
    color.set()
    
    context.setLineWidth(1)
    
    for line in lines {
        
        var startCoords: CGPoint
        var endCoords: CGPoint
        if line.theta == 0 {
            startCoords = CGPoint(x: CGFloat(line.distance), y: 0)
            endCoords = CGPoint(x: CGFloat(line.distance),
                                    y: image.size.height)
        } else if line.theta == 90 {
            startCoords = CGPoint(x: 0, y: CGFloat(line.distance))
            endCoords = CGPoint(x: image.size.width,
                                    y: CGFloat(line.distance))
        } else {
            let rtheta = /*0.0 - */Double(line.theta) * Double.pi / 180
            
            let m = 0.0 - 1.0 / tan(rtheta)
            let b = (Double(line.distance)) / sin(rtheta)
            
            startCoords = CGPoint(x: 0, y: b)
            endCoords = CGPoint(x: image.size.width, y: (m * image.size.width + b))
        }
        
        
        
        context.move(to: startCoords)
        context.addLine(to: endCoords)
        
        context.strokePath()
    }
    
    return UIGraphicsGetImageFromCurrentImageContext()!
}


/*
    Hough Transform
*/

fileprivate func houghSpaceDimensions(image: UIImage) -> CGSize {
    let longestDistance = sqrt(
        pow(image.size.width, 2) + pow(image.size.height, 2)
    )
    return CGSize(width: Int(ceil(longestDistance) * 2), height: 180)
}

fileprivate func loadGrayscaleImage(image: UIImage) -> Matrix {
    let width = Int(image.size.width)
    let height = Int(image.size.height)
    
    var bitmapData = [UInt8](repeating: 0,
                             count: width * height)
    let context = createImageContext(data: &bitmapData,
                                     size: image.size)
    
    context.draw(image.cgImage!, in: CGRect(x: 0, y: 0,
                                            width: width, height: height))
    
    var imageMatrix = emptyMatrix(size: image.size)
    
    for i in 0..<bitmapData.count {
        let row = i / Int(width)
        let col = i % width
        
        imageMatrix[row][col] = Double(bitmapData[i])
    }
    
    return imageMatrix
}

fileprivate func loadMatrix(data: [UInt8], width: CGFloat, height: CGFloat) -> Matrix {
//    let width = Int(image.size.width)
//    let height = Int(image.size.height)
    
//    var bitmapData = [UInt8](repeating: 0,
//                             count: Int(width * height))
//    let context = createImageContext(data: &bitmapData,
//                                     size: image.size)
//
//    context.draw(image.cgImage!, in: CGRect(x: 0, y: 0,
//                                            width: width, height: height))
    
    var imageMatrix = emptyMatrix(size: CGSize(width: width, height: height))
    
    for i in 0..<data.count {
        let row = i / Int(width)
        let col = i % Int(width)
        
        imageMatrix[row][col] = Double(data[i])
    }
    
    return imageMatrix
}

func houghSpace(image: UIImage, data:[UInt8]) -> Matrix {
//    let matrix = loadGrayscaleImage(image: image)
    let matrix = loadMatrix(data: data, width: image.size.width, height: image.size.height)
    print("IMAGE LOADED")
    let dimensions = houghSpaceDimensions(image: image)
    var space = emptyMatrix(size: dimensions)
    
    print("ITERATING")
    for (i, row) in matrix.enumerated() {
        for (j, _) in row.enumerated() {
            let intensityValue = matrix[i][j]
            
            if intensityValue >= CLAMP_THRESHOLD {
                for theta in 0..<180 {
                    let rtheta = Double(theta) * Double.pi / 180
                    let distance =
                        Double(j) * cos(rtheta) + Double(i) * sin(rtheta)
                    
                    space[theta][Int(distance) + Int(dimensions.width) / 2] += 1
                }
            }
        }
    }
    
    return space
}

func topLines(houghSpace space: Matrix, n: Int) -> [Line] {
    let matrixSize = size(ofMatrix: space)
    
    var lines: [Line] = [Line](
        repeating: (theta: 0, distance: 0, occurrence: 0),
        count: Int(matrixSize.width * matrixSize.height)
    )
    
    for (i, row) in space.enumerated() {
        for (j, col) in row.enumerated() {
            let occurrence = space[i][j]
            lines[i * Int(matrixSize.width) + j]  = (
                theta: i,
                distance: j,
                occurrence: Int(occurrence)
            )

        }
    }
    
    let sortedLines = lines.sorted() { a, b in
        return a.occurrence < b.occurrence
    }
    
    return Array<Line>(sortedLines.suffix(n).reversed())
}

func peakLines(houghSpace space: Matrix, n: Int) -> [Line] {
    let matrixSize = size(ofMatrix: space)
    
    var lines: [Line] = []
    let stepSize = 80
    
    var i:Int = 0
    while i < space.count {
        var j:Int = 0
        let row = space[i]
        while j < row.count {
            var sectionMax = -1
            var bestIndex: [Int] = []
            for x in i..<min(space.count, i+stepSize) {
                for y in j..<min(row.count, j+stepSize) {
                    if Int(space[x][y]) > sectionMax {
                        sectionMax = Int(space[x][y])
                        bestIndex = [x, y]
                    }
                }
            }
            lines.append((
                theta: bestIndex[0],
                distance: bestIndex[1] - space[0].count / 2,
                occurrence: sectionMax))
            j += stepSize
        }
        i += stepSize
    }
    
    let sortedLines = lines.sorted() { a, b in
        return a.occurrence < b.occurrence
    }
    
    return Array<Line>(sortedLines.suffix(n).reversed())
}
