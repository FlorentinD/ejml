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

import org.ejml.data.DMatrixSparseCSC;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

@SuppressWarnings({"UnusedMethod"})
public class PageRankTest_DSCC {
    DMatrixSparseCSC inputMatrix;

    PageRank_DSCC pageRank = new PageRank_DSCC();

    @BeforeEach
    public void setUp() {
        inputMatrix = new DMatrixSparseCSC(10, 10, 9);
        inputMatrix.set(1, 2, 1);
        inputMatrix.set(2, 1, 1);
        inputMatrix.set(3, 0, 1);
        inputMatrix.set(3, 1, 1);
        inputMatrix.set(4, 1, 1);
        inputMatrix.set(4, 3, 1);
        inputMatrix.set(5, 1, 1);
        inputMatrix.set(5, 4, 1);
    }

    @Test
    public void pageRank() {
        float tolerance = 1e-7f;
        PageRank_DSCC.PageRankResult result = pageRank.compute(inputMatrix, 0.85, tolerance, 100);

        double[] expected = {
                0.04881240953046283,
                0.37252731373997194,
                0.34566197629884704,
                0.04658515322643894,
                0.04134455015814745,
                0.029013719409226278,
                0.029013719409226278,
                0.029013719409226278,
                0.029013719409226278,
                0.029013719409226278
        };

        assertEquals(88, result.iterations());
        assertArrayEquals(expected, result.result(), tolerance);
    }

    // TODO test other pageRank impl. with a graph with only dangling nodes (compute2)
}
