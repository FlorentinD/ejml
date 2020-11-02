/*
 * Copyright (c) 2009-2020, Peter Abeles. All Rights Reserved.
 *
 * This file is part of Efficient Java Matrix Library (EJML).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ejml.sparse.csc.mult;

import org.ejml.EjmlUnitTests;
import org.ejml.TestUtilEjml;
import org.ejml.UtilEjml;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.data.DVectorSparse;
import org.ejml.data.UtilTestMatrix;
import org.ejml.ops.DSemiRing;
import org.ejml.ops.DSemiRings;
import org.ejml.sparse.csc.CommonOps_DSCC;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Arrays;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@SuppressWarnings({"UnusedMethod"})
public class TestMatrixSparseVectorMultWithSemiRing_DSCC {
    DMatrixSparseCSC inputMatrix;

    @BeforeEach
    public void setUp() {
        // based on example in http://mit.bme.hu/~szarnyas/grb/graphblas-introduction.pdf
        inputMatrix = new DMatrixSparseCSC(7, 7, 12);
        inputMatrix.set(0, 1, 1);
        inputMatrix.set(0, 3, 1);
        inputMatrix.set(1, 4, 1);
        inputMatrix.set(1, 6, 1);
        inputMatrix.set(2, 5, 1);
        inputMatrix.set(3, 0, 0.2);
        inputMatrix.set(3, 2, 0.4);
        inputMatrix.set(4, 5, 1);
        inputMatrix.set(5, 2, 0.5);
        inputMatrix.set(6, 2, 1);
        inputMatrix.set(6, 3, 1);
        inputMatrix.set(6, 4, 1);
        inputMatrix.indicesSorted = true;
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("vectorMatrixMultSources")
    void mult_v_A(String desc, DSemiRing semiRing, double[] expected) {
        // graphblas == following outgoing edges of source nodes
        var v = new DVectorSparse(7);
        v.set(3, 0.5);
        v.set(5, 0.6);
        v.setIndicesSorted(true);

        var found = MatrixSparseVectorMultWithSemiRing_DSCC.mult(v, inputMatrix, null, semiRing);

        CommonOps_DSCC.checkStructure(found.oneDimMatrix);
        assertEquals(2, found.nz_length());
        assertEquals(expected[0], found.get(0));
        assertEquals(expected[1], found.get(2));
    }

//    @ParameterizedTest(name = "{0}")
//    @MethodSource("matrixVectorMultSources")
//    void mult_A_v(String desc, DSemiRing semiRing, double[] expected) {
//        // graphblas == following incoming edges of source nodes
//        double[] v = new double[7];
//        Arrays.fill(v, semiRing.add.id);
//        v[3] = 0.5;
//        v[4] = 0.6;
//
//        double[] found = new double[7];
//
//        MatrixVectorMultWithSemiRing_DSCC.mult(inputMatrix, v, found, semiRing);
//
//        assertTrue(Arrays.equals(found, expected));
//    }

//    @Test
//    void multTransA() {
//        DMatrixSparseCSC transposedMatrix = CommonOps_DSCC.transpose(inputMatrix, null, null);
//
//        double[] v = new double[7];
//        v[3] = 0.5;
//        v[4] = 0.6;
//
//        double[] expected = new double[7];
//        double[] found = new double[7];
//
//        // assuming `mult` works correctly
//        assertTrue(
//                Arrays.equals(
//                        MatrixVectorMultWithSemiRing_DSCC.mult(inputMatrix, v, expected, DSemiRings.PLUS_TIMES, null, null),
//                        MatrixVectorMultWithSemiRing_DSCC.multTransA(transposedMatrix, v, found, DSemiRings.PLUS_TIMES, null, null)
//                )
//        );
//
//    }

    private static Stream<Arguments> vectorMatrixMultSources() {
        return Stream.of(
                Arguments.of("Plus, Times", DSemiRings.PLUS_TIMES, new double[]{0.1, 0.5}),
                Arguments.of("OR, AND", DSemiRings.OR_AND, new double[]{1, 1}),
                Arguments.of("MIN, PLUS", DSemiRings.MIN_PLUS,
                        new double[]{0.7, 0.9}),
                Arguments.of("MAX, PLUS", DSemiRings.MAX_PLUS,
                        new double[]{0.7, 1.1}),
                Arguments.of("MIN, TIMES", DSemiRings.MIN_TIMES,
                        new double[]{0.1, 0.2}),
                Arguments.of("MAX, MIN", DSemiRings.MAX_MIN,
                        new double[]{0.2, 0.5})
        );
    }

    private static Stream<Arguments> matrixVectorMultSources() {
        return Stream.of(
                Arguments.of("PLUS, TIMES", DSemiRings.PLUS_TIMES, new double[]{0.5, 0.6, 1.1}),
                Arguments.of("OR, AND", DSemiRings.OR_AND, new double[]{1, 1, 1}),
                Arguments.of("MIN, PLUS", DSemiRings.MIN_PLUS,
                        new double[]{1.5, 1.6, 1.5})
        );
    }
}
