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

public class PrimitiveDMask extends Mask {
    // TODO: Check if creating dense boolean mask is worth it (currently converting to boolean on the fly)
    private final double[] values;
    // alternatively declare extra separate masks for primitive and dense DMatrix
    private final int numCols;

    public PrimitiveDMask(double[] values, boolean negated) {
        // for dense structures they cannot be used for structural masks
        this(values, 1, negated);
    }

    public PrimitiveDMask(double[] values, int numCols, boolean negated) {
        // for dense structures they cannot be used for structural masks
        super(negated);
        this.values = values;
        this.numCols = numCols;
    }

    @Override
    public boolean isSet(int row, int col) {
        // XOR as negated flips the mask flag
        return negated ^ (values[row * numCols + col] != 0);
    }

    public boolean isSet(int index) {
        return negated ^ (values[index] != 0);
    }
}
