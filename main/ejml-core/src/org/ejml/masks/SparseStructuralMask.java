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

import org.ejml.data.MatrixSparse;

/**
 * only looking if the entry is assigned in the source(disregarding the actual stored value)
 * ! it does not copy the input matrix -> changing the matrix structure will also affect the mask
 */
public class SparseStructuralMask extends Mask {
    private final MatrixSparse matrix;

    public SparseStructuralMask(MatrixSparse matrix, boolean negated) {
        super(negated);
        this.matrix = matrix;
    }

    @Override
    public boolean isSet(int row, int col) {
        return negated ^ matrix.isAssigned(row, col);
    }
}