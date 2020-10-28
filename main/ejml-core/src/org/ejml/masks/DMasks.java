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

import org.ejml.data.*;

/**
 * Helper class to get the corresponding mask builder based on a matrix or primitive array
 */
public class DMasks {
    public static PrimitiveDMask.Builder builder(double[] values) {
        return new PrimitiveDMask.Builder(values);
    }

    public static PrimitiveDMask.Builder builder(DMatrixD1 matrix) {
        return new PrimitiveDMask.Builder(matrix.data).withNumCols(matrix.numCols);
    }

    public static MaskBuilder builder(DMatrixSparseCSC matrix, boolean structural){
        if (structural) {
            return new SparseStructuralMask.Builder(matrix);
        }
        else {
            return new SparseDMask.Builder(matrix);
        }
    }
}
