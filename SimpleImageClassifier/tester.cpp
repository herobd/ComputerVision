
void test()
{
    
    assert(hasPrefix("abcd","a");
    assert(hasPrefix("abcd","abcd");
    assert(!hasPrefix("abcd","bcd");
    assert(!hasPrefix("abcd","abcde");
    assert(!hasPrefix(" abcd","a");
    
    assert(isTrainingImage(1));
    assert(!isTrainingImage(2));
    assert(!isTestingImage(1));
    assert(isTestingImage(2));
    
    assert(0.0 == my_stof(".0"));
    assert(0.0 == my_stof("0.0"));
    assert(0.0 == my_stof("0"));
    assert(10.0 == my_stof("10.0"));
    assert(100.0 == my_stof("1.0e2"));
    assert(0.0001 == my_stof("1.0e-4"));
    
    vector<double> desc0 = {1.0,2.0,3.0};
    struct svm_node* test_node0 = convertDescription(&desc0);
    assert(test_node0[0].value = 1.0);
    assert(test_node0[1].value = 2.0);
    assert(test_node0[2].value = 3.0);
    assert(test_node0[0].index = 0);
    assert(test_node0[1].index = 1);
    assert(test_node0[2].index = 2);
    assert(test_node0[3].index = -1);
    
    vector<double> desc1 = {1.0,0.0,3.0};
    struct svm_node* test_node1 = convertDescription(&desc0);
    assert(test_node1[0].value = 1.0);
    assert(test_node1[1].value = 3.0);
    assert(test_node1[0].index = 0);
    assert(test_node1[1].index = 2);
    assert(test_node1[2].index = -1);
}

