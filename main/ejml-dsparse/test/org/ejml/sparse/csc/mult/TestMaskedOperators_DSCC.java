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

package org.ejml.sparse.csc.mult;

import org.ejml.data.DMatrixSparse;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.masks.DMasks;
import org.ejml.masks.Mask;
import org.ejml.masks.PrimitiveDMask;
import org.ejml.ops.DSemiRing;
import org.ejml.ops.DSemiRings;
import org.ejml.sparse.csc.CommonOpsWithSemiRing_DSCC;
import org.ejml.sparse.csc.CommonOps_DSCC;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Iterator;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

// TODO: proper base test that includes the matrix
public class TestMaskedOperators_DSCC extends BaseTestMatrixMatrixOpsWithSemiRing_DSCC {

    DSemiRing semiRing = DSemiRings.PLUS_TIMES;

    @Test
    public void mult_v_A() {

        // graphblas == following outgoing edges of source nodes
        double v[] = new double[7];
        Arrays.fill(v, semiRing.add.id);
        v[3] = 0.5;
        v[0] = 0.6;

        System.out.println("input vector = " + Arrays.toString(v));

        double[] found = new double[7];
        double[] foundWithMask = found.clone();

        // == dont calculate for existing entries (currently still zero-ing out very likely)
        PrimitiveDMask mask = new PrimitiveDMask(v, true);

        MatrixVectorMultWithSemiRing_DSCC.mult(v, inputMatrix, found, semiRing);
        MatrixVectorMultWithSemiRing_DSCC.mult(v, inputMatrix, foundWithMask, semiRing, mask);

        assertMaskedResult(found, foundWithMask, mask);
    }

    @Test
    public void mult_A_v() {
        // graphblas == following incoming edges of source nodes
        double[] v = new double[7];
        Arrays.fill(v, semiRing.add.id);
        v[3] = 0.5;
        v[4] = 0.6;

        double[] found = new double[7];
        double[] foundWithMask = found.clone();

        // == dont calculate for existing entries (currently still zero-ing out very likely)
        PrimitiveDMask mask = new PrimitiveDMask(v, true);

        MatrixVectorMultWithSemiRing_DSCC.mult(inputMatrix, v, found, semiRing);
        MatrixVectorMultWithSemiRing_DSCC.mult(inputMatrix, v, foundWithMask, semiRing, mask);

        assertMaskedResult(found, foundWithMask, mask);
    }

    // matrix, matrix ops
    // TODO refactor common parts & parameterize masks

    @Test
    public void mult_A_B() {
        // graphblas == following outgoing edges of source nodes
        DMatrixSparseCSC vector = new DMatrixSparseCSC(1, 7);
        vector.set(0, 3, 0.5);
        vector.set(0, 5, 0.6);

        DMatrixSparseCSC found = CommonOpsWithSemiRing_DSCC.mult(vector, inputMatrix, null, semiRing, null);

        // TODO: parameterize test
        boolean negated = true;
        boolean structural = true;
        Mask mask = DMasks.of(vector, negated, structural);

        DMatrixSparseCSC foundWithMask = CommonOpsWithSemiRing_DSCC.mult(vector, inputMatrix, null, semiRing, mask);

        assertMaskedResult(found, foundWithMask, mask);
    }

    @Test
    public void mult_T_A_B() {
        // graphblas == following outgoing edges of source nodes
        DMatrixSparseCSC vector = new DMatrixSparseCSC(1, 7);
        vector.set(0, 3, 0.5);
        vector.set(0, 5, 0.6);

        DMatrixSparseCSC transposed_vector = CommonOps_DSCC.transpose(vector, null, null);

        DMatrixSparseCSC foundTA = CommonOpsWithSemiRing_DSCC.multTransA(transposed_vector, inputMatrix, null, semiRing, null, null, null);

        boolean negated = true;
        boolean structural = true;
        Mask mask = DMasks.of(vector, negated, structural);

        DMatrixSparseCSC foundTAWithMask = CommonOpsWithSemiRing_DSCC.multTransA(transposed_vector, inputMatrix, null, semiRing, mask, null, null);

        assertMaskedResult(foundTA, foundTAWithMask, mask);
    }

    @Test
    public void mult_A_T_B() {
        // graphblas == following outgoing edges of source nodes
        DMatrixSparseCSC vector = new DMatrixSparseCSC(1, 7);
        vector.set(0, 3, 0.5);
        vector.set(0, 5, 0.6);

        DMatrixSparseCSC transposed_matrix = CommonOps_DSCC.transpose(inputMatrix, null, null);

        DMatrixSparseCSC foundTB = CommonOpsWithSemiRing_DSCC.multTransB(vector, transposed_matrix, null, semiRing, null, null, null);

        boolean negated = true;
        boolean structural = true;
        Mask mask = DMasks.of(vector, negated, structural);

        DMatrixSparseCSC foundTBWithMask = CommonOpsWithSemiRing_DSCC.multTransB(vector, transposed_matrix, null, semiRing, mask, null, null);

        assertMaskedResult(foundTB, foundTBWithMask, mask);
    }


    @Test
    public void add_A_B() {
        // graphblas == following outgoing edges of source nodes
        DMatrixSparseCSC vector = new DMatrixSparseCSC(7, 7);
        vector.set(0, 3, 0.5);
        vector.set(0, 5, 0.6);

        DMatrixSparseCSC found = new DMatrixSparseCSC(0, 0);
        DMatrixSparseCSC foundWithMask = found.createLike();

        CommonOpsWithSemiRing_DSCC.add(1, vector, 1, inputMatrix, found, semiRing, null, null, null);

        // TODO: parameterize test
        boolean negated = true;
        boolean structural = true;
        Mask mask = DMasks.of(vector, negated, structural);

        CommonOpsWithSemiRing_DSCC.add(1, vector, 1, inputMatrix, foundWithMask, semiRing, mask, null, null);

        System.out.println("found");
        found.print();

        System.out.println("foundWithMask");
        foundWithMask.print();

        assertMaskedResult(found, foundWithMask, mask);
    }

    // TODO: test elementWise-Mult, apply and reduce



    private void assertMaskedResult(DMatrixSparseCSC found, DMatrixSparseCSC foundWithMask, Mask mask) {
        Iterator<DMatrixSparse.CoordinateRealValue> it = found.createCoordinateIterator();
        it.forEachRemaining(value -> {
            if (mask.isSet(value.row, value.col)) {
                assertEquals(found.get(value.row, value.col), foundWithMask.get(value.row, value.col));
            }
            else {
                // TODO: check if it is the value of the input result matrix (currently just overwritten)
                assertEquals(semiRing.add.id,  foundWithMask.get(value.row, value.col));
            }
        });
    }

    private void assertMaskedResult(double[] found, double[] foundWithMask, PrimitiveDMask mask) {
        for (int i = 0; i < found.length; i++) {
            if (mask.isSet(i)) {
                assertEquals(foundWithMask[i], found[i]);
            } else {
                // at some point this should be v[i] == foundWithMask[i]
                assertEquals(semiRing.add.id, foundWithMask[i]);
            }
        }
    }
}