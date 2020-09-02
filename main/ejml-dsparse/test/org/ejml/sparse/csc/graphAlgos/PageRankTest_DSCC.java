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

import static org.ejml.sparse.csc.graphAlgos.Bfs_DSCC.BfsResult;
import static org.ejml.sparse.csc.graphAlgos.Bfs_DSCC.BfsVariation;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
        PageRank_DSCC.PageRankResult result = pageRank.compute(inputMatrix, 0.85, 40);

        System.out.println("result = " + Arrays.toString(result.result()));
        System.out.println("iterations = " + result.iterations());
    }

    // TODO test other pageRank impl. with a graph with only dangling nodes (compute2)
}
