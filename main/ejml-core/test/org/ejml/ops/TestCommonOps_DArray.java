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

package org.ejml.ops;

import org.ejml.data.DMatrixSparseCSC;
import org.ejml.masks.DMasks;
import org.ejml.masks.Mask;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

public class TestCommonOps_DArray {

    private final double[] values = new double[]{1.5, -1, 30};

    @Test
    void apply() {
        double[] expected = {3.0, -2, 60};

        CommonOps_DArray.apply(values, i -> i * 2);

        assertTrue(Arrays.equals(expected, values));
    }

    @Test
    void applyIdx() {
        double[] expected = {1.5, 0, 32};

        CommonOps_DArray.applyIdx(values, values, (r, i) -> r + i);

        assertTrue(Arrays.equals(expected, values));
    }

    @Test
    void applyAndStoreInOtherArray() {
        double[] expected = {3.0, -2, 60};

        double[] out = new double[expected.length];
        CommonOps_DArray.apply(values, out, i -> i*2);

        assertTrue(Arrays.equals(expected, out));
    }

    @Test
    void elementWiseMult() {
        double[] otherValues = {4, 2, 5};
        double[] output = new double[3];

        CommonOps_DArray.elementWiseMult(values, otherValues, output, Math::min);

        double[] expected = {1.5, -1, 5};
        assertTrue(Arrays.equals(output, expected));
    }

    @Test
    void reduceScalar() {
        assertEquals(-1D ,CommonOps_DArray.reduceScalar(values, 0, DMonoids.MIN.func));
    }

    @Test
    void assignColumnVector() {
        DMatrixSparseCSC sparseVector = new DMatrixSparseCSC(values.length, 1);
        sparseVector.set(0, 0, -42);
        sparseVector.set(2, 0, -1337);

        var output = CommonOps_DArray.assign(values.clone(), sparseVector);

        double[] expected = {-42, values[1], -1337};
        assertArrayEquals(expected, output);
    }

    @Test
    void assignRowVector() {
        DMatrixSparseCSC sparseVector = new DMatrixSparseCSC(1, values.length);
        sparseVector.set(0, 0, -42);
        sparseVector.set(0, 2, -1337);

        var output = CommonOps_DArray.assign(values.clone(), sparseVector);

        double[] expected = {-42, values[1], -1337};
        assertArrayEquals(expected, output);
    }

    @Test
    void assignScalar() {
        DMatrixSparseCSC mask = new DMatrixSparseCSC(1, values.length);
        mask.set(0, 0, 5);
        mask.set(0, 2, 5);

        var output = CommonOps_DArray.assignScalar(values.clone(), 3, mask);

        double[] expected = {3, values[1], 3};
        assertArrayEquals(expected, output);
    }

    @Test
    void assignScalarPrimitive() {
        double[] mask = {1, -1, 1};

        var output = CommonOps_DArray.assignScalar(values.clone(), 3,  DMasks.builder(mask).withZeroElement(-1).build());

        double[] expected = {3, values[1], 3};
        assertArrayEquals(expected, output);
    }
}
