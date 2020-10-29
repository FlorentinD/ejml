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

package org.ejml.masks;

import org.ejml.data.DMatrixSparseCSC;
import org.ejml.sparse.csc.RandomMatrices_DSCC;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestSparseStructuralMasks {

    @Test
    void testSparseStructuralMasks() {
        int dim = 20;
        DMatrixSparseCSC matrix = RandomMatrices_DSCC.rectangle(dim, dim, 5, new Random(42));

        SparseStructuralMask.Builder builder = new SparseStructuralMask.Builder(matrix);
        Mask mask = builder.withNegated(false).build();
        Mask negated_mask = builder.withNegated(true).build();

        for (int row = 0; row < dim; row++) {
            for (int col = 0; col < dim; col++) {
                boolean expected = matrix.isAssigned(row, col);
                assertEquals(mask.isSet(row, col), expected);
                assertEquals(negated_mask.isSet(row, col), !expected);
            }
        }
    }
}
