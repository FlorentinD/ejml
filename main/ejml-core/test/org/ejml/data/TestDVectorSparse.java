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

package org.ejml.data;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestDVectorSparse {
    // TODO: more sophisticated tests

    @Test
    void getterSetter() {
        var v = new DVectorSparse(3);
        v.set(0, 1);
        assertEquals(1, v.get(0));
    }

    @Test
    void append() {
        var v = new DVectorSparse(10, 1);
        v.append(0, 2);
        v.append(2, 3);

        assertEquals(v.nz_length(), 2);
        assertEquals(2 ,v.get(0));
        assertEquals(3 ,v.get(2));
    }

    @Test
    void convertDenseVector() {
        double[] values = {1, 0, 3, 0, 0, 4};

        var v = new DVectorSparse(values, 0);

        assertEquals(6, v.size());
        assertEquals(3, v.nz_length());
        assertEquals(1, v.get(0));
        assertEquals(3, v.get(2));
        assertEquals(4, v.get(5));
    }
}
