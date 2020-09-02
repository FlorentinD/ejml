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

package org.ejml.sparse.csc.graphAlgos;

import org.ejml.EjmlUnitTests;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.DMatrixSparseCSC;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Arrays;
import java.util.stream.Stream;

import static org.ejml.sparse.csc.graphAlgos.Bfs_DSCC.*;
import static org.junit.jupiter.api.Assertions.*;

@SuppressWarnings({"UnusedMethod"})
public class BfsTest_DSCC {
    DMatrixSparseCSC inputMatrix;

    Bfs_DSCC bfs = new Bfs_DSCC();

    private static Stream<Arguments> bfsVariantSource() {

        return Stream.of(
                Arguments.of(BfsVariation.BOOLEAN, new double[] {1,1,1,1,1,1,1}),
                Arguments.of(BfsVariation.LEVEL, new double[] {1,2,3,2,3,4,3}),
                Arguments.of(BfsVariation.PARENTS, new double[] {1,1,4,1,2,3,2})

        );
    }

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
    }

    @ParameterizedTest
    @MethodSource("bfsVariantSource")
    public void testSparseVariations(BfsVariation variation, double[] expected) {
        int[] startNodes = {0};
        int maxIterations = 20;
        Bfs_DSCC.BfsSparseResult result = bfs.computeSparse(inputMatrix, variation, startNodes, maxIterations);

        DMatrixRMaj expectedMatrix = new DMatrixRMaj(1, inputMatrix.numCols, true, expected);

        assertEquals(result.iterations(), 3);
        EjmlUnitTests.assertEquals(expectedMatrix, result.result());
    }

    @ParameterizedTest
    @MethodSource("bfsVariantSource")
    public void testDenseVariations(BfsVariation variation, double[] expected) {
        int startNode = 0;
        int maxIterations = 20;

        Bfs_DSCC.BfsDenseResult result = bfs.computeDense(inputMatrix, variation, startNode, maxIterations);

        assertEquals(result.iterations(), 3);
        assertTrue(Arrays.equals(expected, result.result()));
    }

    @Test
    public void testEmptyResult() {
        BfsResult denseIt = bfs.computeDense(new DMatrixSparseCSC(2, 2), BfsVariation.LEVEL, 0, 5);

        int[] startNodes = {0};
        BfsResult sparseIt = bfs.computeSparse(new DMatrixSparseCSC(2, 2), BfsVariation.LEVEL, startNodes, 5);

        assertEquals(denseIt.iterations(), sparseIt.iterations());
        assertEquals(denseIt.nodesVisited(), sparseIt.nodesVisited());
    }
}