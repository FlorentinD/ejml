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
import org.ejml.ops.DBinaryOperator;
import org.ejml.ops.DSemiRing;
import org.ejml.ops.DSemiRings;
import org.ejml.sparse.csc.CommonOpsWithSemiRing_DSCC;
import org.ejml.sparse.csc.CommonOps_DSCC;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Iterator;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestMaskedOperators_DSCC extends BaseTestMatrixMatrixOpsWithSemiRing_DSCC {

    DSemiRing semiRing = DSemiRings.PLUS_TIMES;

    @Test
    public void mult_v_A() {

        // graphblas == following outgoing edges of source nodes
        double v[] = new double[7];
        Arrays.fill(v, semiRing.add.id);
        v[3] = 0.5;
        v[0] = 0.6;

        double[] prevResult = new double[7];
        prevResult[3] = 99;
        prevResult[0] = 42;
        double[] found = prevResult.clone();
        double[] foundWithMask = prevResult.clone();

        // == dont calculate for existing entries (currently still zero-ing out very likely)
        PrimitiveDMask mask = new PrimitiveDMask(prevResult, true);

        MatrixVectorMultWithSemiRing_DSCC.mult(v, inputMatrix, found, semiRing);
        MatrixVectorMultWithSemiRing_DSCC.mult(v, inputMatrix, foundWithMask, semiRing, mask);

        assertMaskedResult(prevResult, found, foundWithMask, mask);
    }

    @Test
    public void mult_A_v() {
        // graphblas == following incoming edges of source nodes
        double[] v = new double[7];
        Arrays.fill(v, semiRing.add.id);
        v[3] = 0.5;
        v[4] = 0.6;

        double[] prevResult = new double[7];
        prevResult[3] = 99;
        prevResult[0] = 42;
        double[] found = prevResult.clone();
        double[] foundWithMask = prevResult.clone();

        // == dont calculate for existing entries (currently still zero-ing out very likely)
        PrimitiveDMask mask = new PrimitiveDMask(prevResult, true);

        MatrixVectorMultWithSemiRing_DSCC.mult(inputMatrix, v, found, semiRing);
        MatrixVectorMultWithSemiRing_DSCC.mult(inputMatrix, v, foundWithMask, semiRing, mask);

        assertMaskedResult(prevResult, found, foundWithMask, mask);
    }

    // matrix, matrix ops
    // TODO refactor common parts & parameterize masks

    @Test
    public void mult_A_B() {
        // graphblas == following outgoing edges of source nodes
        DMatrixSparseCSC vector = new DMatrixSparseCSC(1, 7);
        vector.set(0, 3, 0.5);
        vector.set(0, 4, 0.6);

        DMatrixSparseCSC prevResult = vector.copy();
        prevResult.set(0, 2, 99);

        DMatrixSparseCSC found = CommonOpsWithSemiRing_DSCC.mult(vector, inputMatrix, prevResult.copy(), semiRing, null);

        // TODO: parameterize test .. also changes expected result
        boolean negated = true;
        boolean structural = true;
        Mask mask = DMasks.of(prevResult, negated, structural);

        DMatrixSparseCSC foundWithMask = CommonOpsWithSemiRing_DSCC.mult(vector, inputMatrix, prevResult.copy(), semiRing, mask);

        assertMaskedResult(prevResult, found, foundWithMask, mask);
    }

    @Test
    public void mult_T_A_B() {
        // graphblas == following outgoing edges of source nodes
        DMatrixSparseCSC vector = new DMatrixSparseCSC(1, 7);
        vector.set(0, 3, 0.5);
        vector.set(0, 5, 0.6);

        DMatrixSparseCSC prevResult = vector.copy();
        prevResult.set(0, 2, 99);


        DMatrixSparseCSC transposed_vector = CommonOps_DSCC.transpose(vector, null, null);

        DMatrixSparseCSC foundTA = CommonOpsWithSemiRing_DSCC.multTransA(transposed_vector, inputMatrix, prevResult.copy(), semiRing, null, null, null);

        boolean negated = true;
        boolean structural = true;
        Mask mask = DMasks.of(prevResult, negated, structural);

        DMatrixSparseCSC foundTAWithMask = CommonOpsWithSemiRing_DSCC.multTransA(transposed_vector, inputMatrix, prevResult.copy(), semiRing, mask, null, null);

        assertMaskedResult(prevResult, foundTA, foundTAWithMask, mask);
    }

    @Test
    public void mult_A_T_B() {
        // graphblas == following outgoing edges of source nodes
        DMatrixSparseCSC inputVector = new DMatrixSparseCSC(1, 7);
        inputVector.set(0, 3, 0.5);
        inputVector.set(0, 5, 0.6);

        DMatrixSparseCSC transposed_matrix = CommonOps_DSCC.transpose(inputMatrix, null, null);

        DMatrixSparseCSC prevResult = inputVector.copy();
        prevResult.set(0, 2, 99);

        DMatrixSparseCSC foundTB = CommonOpsWithSemiRing_DSCC.multTransB(inputVector, transposed_matrix, prevResult.copy(), semiRing, null, null, null);

        boolean negated = true;
        boolean structural = true;
        Mask mask = DMasks.of(prevResult, negated, structural);

        DMatrixSparseCSC foundTBWithMask = CommonOpsWithSemiRing_DSCC.multTransB(inputVector, transposed_matrix, prevResult.copy(), semiRing, mask, null, null);

        assertMaskedResult(prevResult, foundTB, foundTBWithMask, mask);
    }


    @Test
    public void add_A_B() {
        // graphblas == following outgoing edges of source nodes
        DMatrixSparseCSC otherMatrix = new DMatrixSparseCSC(7, 7);
        otherMatrix.set(0, 3, 0.5);
        otherMatrix.set(0, 5, 0.6);

        DMatrixSparseCSC resultInput = new DMatrixSparseCSC(7, 7);
        // these should not be kept in the result (as negated mask)
        resultInput.set(0,0, 99);
        resultInput.set(0,3, 42);

        DMatrixSparseCSC found = CommonOpsWithSemiRing_DSCC.add(1, otherMatrix, 1, inputMatrix, resultInput.copy(), semiRing, null, null, null);

        boolean negated = true;
        boolean structural = true;
        Mask mask = DMasks.of(resultInput, negated, structural);

        DMatrixSparseCSC foundWithMask = CommonOpsWithSemiRing_DSCC.add(1, otherMatrix, 1, inputMatrix, resultInput.copy(), semiRing, mask, null, null);

        assertMaskedResult(resultInput, found, foundWithMask, mask);
    }

    @Test
    public void elementWiseMult() {
        // graphblas == following outgoing edges of source nodes
        DMatrixSparseCSC otherMatrix = new DMatrixSparseCSC(7, 7);
        otherMatrix.set(0, 3, 0.5);
        otherMatrix.set(0, 5, 0.6);

        DMatrixSparseCSC resultInput = new DMatrixSparseCSC(7, 7);
        // these should not be kept in the result (as negated mask)
        resultInput.set(0,0, 99);
        resultInput.set(0,3, 42);

        DMatrixSparseCSC found = CommonOpsWithSemiRing_DSCC.elementMult(otherMatrix, inputMatrix, resultInput.copy(), semiRing, null, null, null);

        boolean negated = true;
        boolean structural = true;
        Mask mask = DMasks.of(resultInput, negated, structural);

        DMatrixSparseCSC foundWithMask = CommonOpsWithSemiRing_DSCC.elementMult(otherMatrix, inputMatrix, resultInput.copy(), semiRing, mask, null, null);

        assertMaskedResult(resultInput, found, foundWithMask, mask);
    }

    @Test
    public void apply() {
        DMatrixSparseCSC inputVector = new DMatrixSparseCSC(1, 7);
        inputVector.set(0, 3, 0.5);
        inputVector.set(0, 5, 0.6);

        DMatrixSparseCSC prevResult = inputVector.copy();
        prevResult.set(0, 2, 99);

        boolean negated = true;
        boolean structural = true;
        Mask mask = DMasks.of(prevResult, negated, structural);

        DBinaryOperator first = (x, y) -> x;
        DMatrixSparseCSC result = CommonOps_DSCC.apply(inputVector, a -> a * 2, prevResult.copy(), null, null);
        DMatrixSparseCSC resultWithMask = CommonOps_DSCC.apply(inputVector, a -> a * 2, prevResult.copy(), mask, first);

        assertMaskedResult(prevResult, result, resultWithMask, mask);
    }

    // TODO: test reduce as well as general test of MaskUtil



    private void assertMaskedResult(DMatrixSparseCSC prevResult, DMatrixSparseCSC found, DMatrixSparseCSC foundWithMask, Mask mask) {
        Iterator<DMatrixSparse.CoordinateRealValue> it = found.createCoordinateIterator();
        // check that existing result were not overwritten
        it.forEachRemaining(value -> {
            if (mask.isSet(value.row, value.col)) {
                assertEquals(found.get(value.row, value.col), foundWithMask.get(value.row, value.col), "Field should have been computed");
            }
            else {
                assertEquals(prevResult.get(value.row, value.col),  foundWithMask.get(value.row, value.col), "Field from initial result was overwritten");
            }
        });

        // checking that untouched cells are still present
        prevResult.createCoordinateIterator().forEachRemaining(value -> {
            if (!mask.isSet(value.row, value.col)) {
                assertEquals(prevResult.get(value.row, value.col),  foundWithMask.get(value.row, value.col), "Field from initial result was deleted");
            }
        });
    }

    private void assertMaskedResult(double[] prevResult, double[] found, double[] foundWithMask, PrimitiveDMask mask) {
        for (int i = 0; i < found.length; i++) {
            if (mask.isSet(i)) {
                assertEquals(foundWithMask[i], found[i], "Computation differs");
            } else {
                assertEquals(prevResult[i], foundWithMask[i], "Initial result was overwritten");
            }
        }
    }
}