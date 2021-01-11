/*
 * Copyright (c) 2020, Peter Abeles. All Rights Reserved.
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

package org.ejml.sparse.csc;

import org.ejml.EjmlUnitTests;
import org.ejml.data.DVectorSparse;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestCommonVectorOps_DSCC {
    @Test
    void addInPlaceVector() {
        Random rand = new Random(42);
        var v = new DVectorSparse(RandomMatrices_DSCC.generateUniform(10, 1, 6, 1, 1, rand));
        var u = new DVectorSparse(RandomMatrices_DSCC.generateUniform(10, 1, 6, 1, 1, rand));

        var result = CommonVectorOps_DSCC.add(u.copy(), v, Double::sum, null).oneDimMatrix;
        var expected = CommonOps_DSCC.add(1, u.oneDimMatrix, 1, v.oneDimMatrix, null, null, null);

        EjmlUnitTests.assertEquals(result, expected);
    }

    @Test
    void assignScalar() {
        Random rand = new Random(42);
        int vectorSize = 10;
        var entriesToAssign = new DVectorSparse(RandomMatrices_DSCC.generateUniform(vectorSize, 1, 6, 1, 1, rand));
        var inputVector = new DVectorSparse(RandomMatrices_DSCC.generateUniform(vectorSize, 1, 6, 1, 1, rand));

        int scalarToAssign = 42;
        var result = CommonVectorOps_DSCC.assignScalar(inputVector.copy(), scalarToAssign, entriesToAssign, null);

        for (int idx = 0; idx < vectorSize; idx++) {
            var resultEntry = result.get(idx);
            if (entriesToAssign.isAssigned(idx)) {
                assertEquals(scalarToAssign, resultEntry);
            } else {
                assertEquals(inputVector.get(idx, Double.NaN), result.get(idx, Double.NaN));
            }
        }
    }
}
