//
//  QuarticSolver.swift
//  ARDepthDataExample
//
//  Created by David McAllister on 5/10/22.
//  Copyright © 2022 Apple. All rights reserved.
//

//import Foundation
//
//
//let NEAR_ZERO: Double = 0.0000001;
//
//class QuarticSolver {
//
//    fileprivate var a: Double = 0;
//    fileprivate var b: Double = 0;
//    fileprivate var c: Double = 0;
//    fileprivate var d: Double = 0;
//    fileprivate var e: Double = 0;
//
//    init(a: Double, b: Double, c: Double, d: Double, e: Double) {
//        self.a = a;
//        self.b = b;
//        self.c = c;
//        self.d = d;
//        self.e = e;
//    }
//
//    func findRealRoots() -> [Double] {
//        return solveUsingFerrariMethodWikipedia();
//    }
//
//    func solveUsingFerrariMethodWikipedia() -> [Double] {
//        // http://en.wikipedia.org/wiki/Quartic_function#Ferrari.27s_solution
//        let depressedQuartic = toDepressed();
////        if (depressedQuartic.isBiquadratic()) {
////            double[] depressedRoots = depressedQuartic.solveUsingBiquadraticMethod();
////            return reconvertToOriginalRoots(depressedRoots);
////        }
//
//        let y = findFerraryY(depressedQuartic);
//        let originalRootConversionPart = -b / (4.0 * a);
//        let firstPart = sqrt(depressedQuartic.c + 2.0 * y);
//
//        double positiveSecondPart = Math.sqrt(-(3.0 * depressedQuartic.c + 2.0 * y + 2.0 * depressedQuartic.d
//                / Math.sqrt(depressedQuartic.c + 2.0 * y)));
//        double negativeSecondPart = Math.sqrt(-(3.0 * depressedQuartic.c + 2.0 * y - 2.0 * depressedQuartic.d
//                / Math.sqrt(depressedQuartic.c + 2.0 * y)));
//
//        double x1 = originalRootConversionPart + (firstPart + positiveSecondPart) / 2.0;
//        double x2 = originalRootConversionPart + (-firstPart + negativeSecondPart) / 2.0;
//        double x3 = originalRootConversionPart + (firstPart - positiveSecondPart) / 2.0;
//        double x4 = originalRootConversionPart + (-firstPart - negativeSecondPart) / 2.0;
//
//        Set<Double> realRoots = findOnlyRealRoots(x1, x2, x3, x4);
//        return toDoubleArray(realRoots);
//    }
//
//    private double[] reconvertToOriginalRoots(double[] depressedRoots) {
//        double[] originalRoots = new double[depressedRoots.length];
//        for (int i = 0; i < depressedRoots.length; ++i) {
//            originalRoots[i] = depressedRoots[i] - b / (4.0 * a);
//        }
//        return originalRoots;
//    }
//
//    func findFerraryY(depressedQuartic: QuarticSolver) -> Double {
//        let a3 = 1.0;
//        let a2 = 5.0 / 2.0 * depressedQuartic.c;
//        let a1 = 2.0 * pow(depressedQuartic.c, 2.0) - depressedQuartic.e;
//        let a0 = pow(depressedQuartic.c, 3.0) / 2.0 - depressedQuartic.c * depressedQuartic.e / 2.0
//                - pow(depressedQuartic.d, 2.0) / 8.0;
//
//        CubicFunction cubicFunction = new CubicFunction(a3, a2, a1, a0);
//        double[] roots = cubicFunction.findRealRoots();
//
//        for (double y : roots) {
//            if (depressedQuartic.c + 2.0 * y != 0.0) {
//                return y;
//            }
//        }
//        throw new IllegalStateException("Ferrari method should have at least one y");
//    }
//
//    func solveUsingBiquadraticMethod() -> [Double] {
//        QuadraticFunction quadraticFunction = new QuadraticFunction(a, c, e);
//        if (!quadraticFunction.hasRoots()) {
//            return new double[] {};
//        }
//
//        double[] quadraticRoots = quadraticFunction.findRoots();
//        Set<Double> roots = new HashSet<>();
//        for (double quadraticRoot : quadraticRoots) {
//            if (quadraticRoot > 0.0) {
//                roots.add(Math.sqrt(quadraticRoot));
//                roots.add(-Math.sqrt(quadraticRoot));
//            } else if (quadraticRoot == 0.00) {
//                roots.add(0.00);
//            }
//        }
//
//        return toDoubleArray(roots);
//    }
//
//    func toDepressed() -> QuarticSolver {
////         http://en.wikipedia.org/wiki/Quartic_function#Converting_to_a_depressed_quartic
//        let p = (8.0 * a * c - 3.0 * pow(b, 2.0)) / (8.0 * pow(a, 2.0));
//        let q = (pow(b, 3.0) - 4.0 * a * b * c + 8.0 * d * pow(a, 2.0)) / (8.0 * pow(a, 3.0));
//        let r = (-3.0 * pow(b, 4.0) + 256.0 * e * pow(a, 3.0) - 64.0 * d * b * pow(a, 2.0) + 16.0 * c
//                * a * pow(b, 2.0))
//                / (256.0 * pow(a, 4.0));
//        return QuarticSolver(a: 1.0, b: 0.0, c: p, d: q, e: r);
//    }
//
//
//    private Set<Double> findOnlyRealRoots(double... roots) {
//        Set<Double> realRoots = new HashSet<>();
//        for (double root : roots) {
//            if (Double.isFinite(root)) {
//                realRoots.add(root);
//            }
//        }
//        return realRoots;
//    }
//
//
//    private double[] toDoubleArray(Collection<Double> values) {
//        double[] doubleArray = new double[values.size()];
//        int i = 0;
//        for (double value : values) {
//            doubleArray[i] = value;
//            ++i;
//        }
//        return doubleArray;
//    }
//}
