//
//  LinearRegression.swift
//  ARDepthDataExample
//
//  Created by David McAllister on 5/9/22.
//  Copyright © 2022 Apple. All rights reserved.
//

import Foundation

fileprivate func average(_ input: [Double]) -> Double {
    return input.reduce(0, +) / Double(input.count)
}

fileprivate func multiply(_ a: [Double], _ b: [Double]) -> [Double] {
    return zip(a,b).map(*)
}

func linearRegression(_ xs: [Double], _ ys: [Double]) -> (Double) -> Double {
    let sum1 = average(multiply(ys, xs)) - average(xs) * average(ys)
    let sum2 = average(multiply(xs, xs)) - pow(average(xs), 2)
    let slope = sum1 / sum2
    let intercept = average(ys) - slope * average(xs)
    return { x in intercept + slope * x }
}
