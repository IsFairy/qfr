/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "gtest/gtest.h"
#include <random>

#include "QuantumComputation.hpp"

class DDFunctionality : public testing::TestWithParam<unsigned short> {
    protected:
        void TearDown() override {
            if (!dd->isTerminal(e))
                dd->decRef(e);
            dd->garbageCollect(true);

            // number of complex table entries after clean-up should equal initial number of entries
            EXPECT_EQ(dd->cn.count, initialComplexCount);
            // number of available cache entries after clean-up should equal initial number of entries
            EXPECT_EQ(dd->cn.cacheCount, initialCacheCount);
        }

        void SetUp() override {
            // dd 
            dd                  = std::make_unique<dd::Package>();
            initialCacheCount   = dd->cn.cacheCount;
            initialComplexCount = dd->cn.count;
			dd->setMode(dd::Matrix);

            // initial state preparation
            line.fill(qc::LINE_DEFAULT);
            e = ident = dd->makeIdent(0, (short)(nqubits-1));
            dd->incRef(ident);

	        std::array<std::mt19937_64::result_type , std::mt19937_64::state_size> random_data{};
	        std::random_device rd;
	        std::generate(begin(random_data), end(random_data), [&](){return rd();});
	        std::seed_seq seeds(begin(random_data), end(random_data));
	        mt.seed(seeds);
	        dist = std::uniform_real_distribution<fp> (0.0, 2 * qc::PI);
        }

        unsigned short                          nqubits             = 4;
        long                                    initialCacheCount   = 0;
        unsigned int                            initialComplexCount = 0;
        std::array<short, qc::MAX_QUBITS>       line{};
        dd::Edge                                e{}, ident{};
        std::unique_ptr<dd::Package>            dd;
		std::mt19937_64                         mt;
		std::uniform_real_distribution<fp>      dist;
};

INSTANTIATE_TEST_SUITE_P(Parameters,
                         DDFunctionality,
                         testing::Values(qc::I, qc::H, qc::X, qc::Y, qc::Z, qc::S, qc::Sdag, qc::T, qc::Tdag, qc::V,
                                         qc::Vdag, qc::U3, qc::U2, qc::Phase, qc::RX, qc::RY, qc::RZ, qc::Peres, qc::Peresdag,
                                         qc::SWAP, qc::iSWAP),
                         [](const testing::TestParamInfo<DDFunctionality::ParamType>& info) {
                             auto gate = (qc::OpType)info.param;
	                         switch (gate) {
                                case qc::I:     return "i";
                                case qc::H:     return "h";
                                case qc::X:     return "x";
                                case qc::Y:     return "y";
                                case qc::Z:     return "z";
                                case qc::S:     return "s";
                                case qc::Sdag:  return "sdg";
                                case qc::T:     return "t";
                                case qc::Tdag:  return "tdg";
                                case qc::V:     return "v";
                                case qc::Vdag:  return "vdg";
                                case qc::U3:    return "u3";
                                case qc::U2:    return "u2";
                                case qc::Phase: return "u1";
                                case qc::RX:    return "rx";
                                case qc::RY:    return "ry";
                                case qc::RZ:    return "rz";
                                case qc::SWAP:  return "swap";
                                case qc::iSWAP: return "iswap";
                                case qc::Peres: return "p";
                                case qc::Peresdag:  return "pdag";
                                default:        return "unknownGate";
                            }
                         }); 


TEST_P(DDFunctionality, standard_op_build_inverse_build) {
    auto gate = (qc::OpType)GetParam();
    
    qc::StandardOperation op;
    switch(gate) {
    	case qc::U3:
		    op = qc::StandardOperation(nqubits, 0,  gate, dist(mt), dist(mt), dist(mt));
		    break;
        case qc::U2: 
            op = qc::StandardOperation(nqubits, 0,  gate, dist(mt), dist(mt));
			break;
        case qc::RX:
		case qc::RY:
        case qc::RZ:
		case qc::Phase:
            op = qc::StandardOperation(nqubits, 0,  gate, dist(mt));
            break;

        case qc::SWAP:
        case qc::iSWAP:
            op = qc::StandardOperation(nqubits, std::vector<unsigned short>{0, 1},  gate);
            break;
        case qc::Peres:
		case qc::Peresdag:
            op = qc::StandardOperation(nqubits, std::vector<qc::Control>{qc::Control(0)}, 1, 2, gate);
            break;
        default:
            op = qc::StandardOperation(nqubits, 0,  gate);
    }

    ASSERT_NO_THROW({e = dd->multiply(op.getDD(dd, line), op.getInverseDD(dd, line));});
    dd->incRef(e);

    EXPECT_TRUE(dd::Package::equals(ident, e));
}

TEST_F(DDFunctionality, build_circuit) {
    qc::QuantumComputation qc(nqubits);

    qc.emplace_back<qc::StandardOperation>(nqubits, 0,  qc::X);
    qc.emplace_back<qc::StandardOperation>(nqubits, std::vector<unsigned short>{0, 1},  qc::SWAP);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0,  qc::H);
    qc.emplace_back<qc::StandardOperation>(nqubits, 3,  qc::S);
    qc.emplace_back<qc::StandardOperation>(nqubits, 2,  qc::Sdag);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0,  qc::V);
    qc.emplace_back<qc::StandardOperation>(nqubits, 1,  qc::T);
    qc.emplace_back<qc::StandardOperation>(nqubits, qc::Control(0), 1,  qc::X);
    qc.emplace_back<qc::StandardOperation>(nqubits, qc::Control(3), 2,  qc::X);
    qc.emplace_back<qc::StandardOperation>(nqubits, std::vector<qc::Control>({qc::Control(3), qc::Control(2)}), 0,  qc::X);
    
    qc.emplace_back<qc::StandardOperation>(nqubits, std::vector<qc::Control>({qc::Control(3), qc::Control(2)}), 0,  qc::X);
    qc.emplace_back<qc::StandardOperation>(nqubits, qc::Control(3), 2,  qc::X);
    qc.emplace_back<qc::StandardOperation>(nqubits, qc::Control(0), 1,  qc::X);
    qc.emplace_back<qc::StandardOperation>(nqubits, 1,  qc::Tdag);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0,  qc::Vdag);
    qc.emplace_back<qc::StandardOperation>(nqubits, 2,  qc::S);
    qc.emplace_back<qc::StandardOperation>(nqubits, 3,  qc::Sdag);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0,  qc::H);
    qc.emplace_back<qc::StandardOperation>(nqubits, std::vector<unsigned short>{0, 1},  qc::SWAP);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0,  qc::X);


    e = dd->multiply(qc.buildFunctionality(dd), e);
   
    //dd->printVector(e);
    //dd->printVector(ident);

    EXPECT_TRUE(dd::Package::equals(ident, e));

    qc.emplace_back<qc::StandardOperation>(nqubits, 0,  qc::X);
    e = dd->multiply(qc.buildFunctionality(dd), e);
    dd->incRef(e);

    EXPECT_FALSE(dd::Package::equals(ident, e));
}

TEST_F(DDFunctionality, non_unitary) {
	qc::QuantumComputation qc;
	auto dummy_map = std::map<unsigned short, unsigned short>{};
	auto op = qc::NonUnitaryOperation(nqubits, {0,1,2,3}, {0,1,2,3});
	EXPECT_FALSE(op.isUnitary());
	try {
		op.getDD(dd, line);
		FAIL() << "Nothing thrown. Expected qc::QFRException";
	} catch (qc::QFRException const & err) {
		std::cout << err.what() << std::endl;
		SUCCEED();
	} catch (...) {
		FAIL() << "Expected qc::QFRException";
	}
	try {
		op.getInverseDD(dd, line);
		FAIL() << "Nothing thrown. Expected qc::QFRException";
	} catch (qc::QFRException const & err) {
		std::cout << err.what() << std::endl;
		SUCCEED();
	} catch (...) {
		FAIL() << "Expected qc::QFRException";
	}
	try {
		op.getDD(dd, line, dummy_map);
		FAIL() << "Nothing thrown. Expected qc::QFRException";
	} catch (qc::QFRException const & err) {
		std::cout << err.what() << std::endl;
		SUCCEED();
	} catch (...) {
		FAIL() << "Expected qc::QFRException";
	}
	try {
		op.getInverseDD(dd, line, dummy_map);
		FAIL() << "Nothing thrown. Expected qc::QFRException";
	} catch (qc::QFRException const & err) {
		std::cout << err.what() << std::endl;
		SUCCEED();
	} catch (...) {
		FAIL() << "Expected qc::QFRException";
	}
}
 
TEST_F(DDFunctionality, GRCSExport) {
	qc::QuantumComputation qc;
    qc.import("circuits/grcs/inst_4x4_80_9_v2.txt");
    e = qc.simulate(dd->makeZeroState(qc.getNqubits()), dd);

    dd::serialize(e, "inst_4x4_80_9_v2_serialized.txt", true);
    dd::Edge result = dd::deserialize(dd, "inst_4x4_80_9_v2_serialized.txt");
    /*
    bool success = dd->equals(e, result);
    if(!success) {
        unsigned short nqubits = qc.getNqubits();
        std::map<std::string, dd::ComplexValue> result_amplitudes;
        std::map<std::string, dd::ComplexValue> ref_amplitudes;

        std::string elements(nqubits, '0');
        dd->getAllAmplitudes(e, result_amplitudes, nqubits - 1, elements);
        dd->getAllAmplitudes(result, ref_amplitudes, nqubits - 1, elements);
        success = dd->compareAmplitudes(ref_amplitudes, result_amplitudes, false);
    }
    EXPECT_TRUE(success);
    */
    EXPECT_TRUE(dd->equals(e, result));
}

TEST_F(DDFunctionality, SupremacyExport) {
    nqubits = 7;
    qc::QuantumComputation quantumComputation(nqubits);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 0, qc::H);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 1, qc::H);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 2, qc::H);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 3, qc::H);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 4, qc::H);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 5, qc::H);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 6, qc::H);

    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 0, qc::T);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 1, qc::T);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 2, qc::T);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 3, qc::T);

    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, qc::Control(4), 5, qc::Z);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, qc::Control(6), 2, qc::Z);

    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 3, qc::RX, qc::PI_2);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, qc::Control(0), 1, qc::Z);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 2, qc::RX, qc::PI_2);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 4, qc::RY, qc::PI_2);

    
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 5, qc::T);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 6, qc::RX, qc::PI_2);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 0, qc::RY, qc::PI_2);

    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, qc::Control(4), 1, qc::Z);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 4, qc::RY, qc::PI_2);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 5, qc::RY, qc::PI_2);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 6, qc::RY, qc::PI_2);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 4, qc::T, qc::PI_2);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 5, qc::T, qc::PI_2);
    quantumComputation.emplace_back<qc::StandardOperation>(nqubits, 6, qc::T, qc::PI_2);
    
    e = quantumComputation.simulate(dd->makeZeroState(quantumComputation.getNqubits()), dd);

    dd::serialize(e, "SupremacyExport.txt", true);
    dd->export2Dot(e, "SupremacyExport_orig", true);
    dd::Edge result = dd::deserialize(dd, "SupremacyExport.txt");
    dd->export2Dot(result, "SupremacyExport_deserialized", true);
    
    EXPECT_TRUE(dd->equals(e, result));
    
    /*
    std::string test = "3 2 (2 0.7071067811865476) () (2 0.90i) ()";		
	std::string complex_real_regex = "([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?(?![ \\d\\.]*(?:[eE][+-])?\\d*[iI]))?";
    std::string complex_imag_regex = "( ?[+-]? ?(?:(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?)?[iI])?";
    std::string edge_regex = " \\(((-?\\d+) (" + complex_real_regex + complex_imag_regex + "))?\\)";
	std::regex complex_weight_regex (complex_real_regex + complex_imag_regex);
    std::regex line_regex ("(\\d+) (\\d+)(?:" + edge_regex + ")(?:" + edge_regex + ")(?:" + edge_regex + ")(?:" + edge_regex + ") *(?:#.*)?");
    std::smatch m;
    
    if(!std::regex_match(test, m, line_regex)) {
		std::cerr << "Regex did not match" << std::endl;
    } else {
		std::cerr << "Regex did match" << std::endl;
        for(int edge_idx = 3, i = 0; i < dd::NEDGE; i++, edge_idx += 5) {
				if(m.str(edge_idx).empty()) {
					// std::cout << "index " << i << " is empty " << std::endl;
					continue;
				}

                std::cout << "real " << m.str(edge_idx + 3) << std::endl;
                std::cout << "imag " << m.str(edge_idx + 4) << std::endl;
			}
    }
    */
}