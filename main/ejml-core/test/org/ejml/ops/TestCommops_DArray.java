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

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestCommops_DArray {

    private final double[] values = new double[]{1.5, -1, 30};

    @Test
    void apply() {
        double[] expected = {3.0, -2, 60};

        CommonOps_DArray.apply(values, i -> i * 2);

        assertTrue(Arrays.equals(expected, values));
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
        assertEquals(-1D ,CommonOps_DArray.reduceScalar(values, DMonoids.MIN));
    }
}