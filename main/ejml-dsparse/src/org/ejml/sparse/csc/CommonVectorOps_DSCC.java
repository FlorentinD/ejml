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

import org.ejml.MatrixDimensionException;
import org.ejml.data.DVectorSparse;
import org.ejml.data.IGrowArray;
import org.ejml.ops.DBinaryOperator;
import org.jetbrains.annotations.Nullable;

import static org.ejml.UtilEjml.adjust;
import static org.ejml.UtilEjml.stringShapes;

public class CommonVectorOps_DSCC {
    /**
     * Performs in-place vector addition:<br>
     * u += v
     *
     * @param u           Matrix
     * @param v           Matrix
     * @param add         Binary operator to define `+`
     * @param gw          (Optional) Storage for internal workspace.  Can be null.
     */
    public static DVectorSparse add( DVectorSparse u, DVectorSparse v, DBinaryOperator add, @Nullable IGrowArray gw) {
        // no mask, as inPlace is not possible if entries of u should not be computed
        // no accumulator .. as inPlace add

        if (u.size() != v.size())
            throw new MatrixDimensionException("Inconsistent vector shapes. " + stringShapes(u, v));

        u.setIndicesSorted(false);

        int[] w = adjust(gw, u.size());

        // mapping vector-index -> nz-index
        for (int i = 0; i < u.nz_length(); i++) {
            w[u.nz_indices()[i]] = i + 1;
        }

        for (int i = 0; i < v.nz_length(); i++) {
            int vIndex = v.nz_indices()[i];
            double v_value = v.nz_values()[i];
            int u_nz_index = w[vIndex] - 1;
            if (u_nz_index >= 0) {
                u.nz_values()[u_nz_index] = add.apply(v_value, u.nz_values()[u_nz_index]);
            } else {
                u.append(vIndex, v_value);
            }
        }

        return u;
    }

    /**
     * Simple in-place assignScalar
     *
     */
    public static DVectorSparse assignScalar( DVectorSparse u, double scalar, DVectorSparse entriesToAssign, @Nullable IGrowArray gw) {

        // not using Mask object here as no iterator atm (hence also called entriesToAssign)
        // also missing accumulator and replace flag

        u.setIndicesSorted(false);

        int[] w = adjust(gw, u.size());

        // mapping vector-index -> nz-index
        for (int i = 0; i < u.nz_length(); i++) {
            w[u.nz_indices()[i]] = i + 1;
        }

        for (int i = 0; i < entriesToAssign.nz_length(); i++) {
            int maskIndex = entriesToAssign.nz_indices()[i];
            int u_nz_index = w[maskIndex] - 1;
            if (u_nz_index >= 0) {
                u.nz_values()[u_nz_index] = scalar;
            } else {
                u.append(maskIndex, scalar);
            }
        }

        return u;
    }
}
