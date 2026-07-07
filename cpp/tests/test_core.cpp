#include "asterion/core.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <array>
#include <limits>

TEST_CASE("host schedule rejects illegal mapping and handles row bounds") {
    std::array<uint64_t,2> shape{17,129};
    std::array<uint64_t,10> fields{256,4,4,32,32,32,1,1,1,0};
    REQUIRE(asterion_abi_version()==2);
    REQUIRE(asterion_verify_schedule(3,shape.data(),2,fields.data(),10)==0);
    fields[1]=3;
    REQUIRE(asterion_verify_schedule(3,shape.data(),2,fields.data(),10)!=0);
    fields[1]=4;
    shape[1]=32769;
    REQUIRE(asterion_verify_schedule(3,shape.data(),2,fields.data(),10)!=0);
}
TEST_CASE("fractional weighted tail and invalid input are handled") {
    const std::array<double,2> values{1,10}, weights{.9,.1};
    double output=-1;
    REQUIRE(asterion_cvar(values.data(),weights.data(),2,.8,&output)==0);
    REQUIRE(output==Catch::Approx(5.5));
    REQUIRE(asterion_cvar(nullptr,weights.data(),2,.8,&output)!=0);
    REQUIRE(asterion_cvar(values.data(),weights.data(),2,1,&output)!=0);
    const std::array<double,2> invalid{1,std::numeric_limits<double>::infinity()};
    REQUIRE(asterion_cvar(invalid.data(),weights.data(),2,.8,&output)!=0);
}
