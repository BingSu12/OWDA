import theano.tensor as T
import theano
import numpy as np
from theano.compile.ops import as_op


@as_op(itypes=[theano.tensor.ivector],
       otypes=[theano.tensor.ivector])
def numpy_unique(a):
    return np.unique(a)

@as_op(itypes=[theano.tensor.lscalar],
       otypes=[theano.tensor.lvector])
def numpy_arange(a):
    return np.arange(a)


def owda_loss(n_components, margin, Yinter):
    """
    The main loss function (inner_lda_objective) is wrapped in this function due to
    the constraints imposed by Keras on objective functions
    """
    def inner_owda_objective(y_true, y_pred):
        """
        It is the loss function of LDA as introduced in the original paper. 
        It is adopted from the the original implementation in the following link:
        https://github.com/CPJKU/deep_lda
        Note: it is implemented by Theano tensor operations, and does not work on Tensorflow backend
        """
        
        r = 1e-4

        # init groups
        L = y_true.shape[1]-1
        dim = y_pred.shape[1]
        N = y_pred.shape[0]
        yt = T.cast(y_true[:,L].flatten(), "int32")
        #groups = numpy_unique(yt)
        classes = Yinter.shape[0]  #groups.length()
        #qest = T.arange(classes)[0]
        #print(qest.eval())
        #print(T.arange(classes).eval())
        #print(yt.eval())
        #print(classes)
        #print(dim)
        #print(L)
        #print(T.eq(yt, qest).eval())
        #print(yt[qest].eval())
        #print(Yinter[0,0,:,:])


        #barycenter = T.tensor3()  #(classes,L,dim)
        #Sw_t = T.zeros(dim,dim)
        #prior = T.zeros(1,classes)
        #for c in range(classes):
        #    Xgt = y_pred[T.eq(yt, c).nonzero()[0], :]
        #    Ygt = y_true[T.eq(yt, c).nonzero()[0], :]
        #    n = Xgt.shape[0]            
        #    prior[c] = n/N
        #    for i in range(n):
        #        for j in range(L):
        #            barycenter[c,j,:] = barycenter[c,j,:] + Xgt[i,:]*Ygt[i,j]/n
        #    conv = T.zeros(dim,dim)
        #    for i in range(n):
        #        for j in  range(L):
        #            temp = Xgt[i,:]-barycenter[c,j,:]
        #            conv = conv + Ygt[i,j]*T.dot(temp.T,temp)/n
        #    Sw_t = Sw_t + prior[c]*conv
        #
        # Sb_t = T.zeros(dim,dim)
        # for c_1 in classes:
        #    for c_2 in range(c_1+1,classes):
        #        for i in range(L):
        #            for j in range(L):
        #                temp = barycenter[c_1,i,:] - barycenter[c_2,j,:]
        #                Sb_t = Sb_t + prior[c_1]*prior[c_2]*Yinter[c_1,c_2,i,j]*T.dot(temp.T,temp)


        def compute_barycenter(group,Xt,yt,y_true,L):
            Xgt = Xt[T.eq(yt, group).nonzero()[0], :]
            Ygt = y_true[T.eq(yt, group).nonzero()[0], :]
            n = Xgt.shape[0]
            #dim = Xt.shape[1]
            
            #print(n)
            #print(dim.eval())
            #print(Xgt.shape.eval())
            #print(T.arange(n))

            def compute_outer(i,Xgt,Ygtv,n):
                n = T.cast(n,'float32')
                return Xgt[i,:]*Ygtv[i]/n
            def compute_outer_n(l,Xgt,Ygt,n):
                Ygtv = Ygt[:,l]
                outpro,updates_1 = theano.scan(fn=compute_outer, outputs_info=None,sequences=numpy_arange(n),non_sequences=[Xgt,Ygtv,n])
                return T.sum(outpro, axis=0)
            outproL,updates_2 = theano.scan(fn=compute_outer_n, outputs_info=None,sequences=T.arange(L),non_sequences=[Xgt,Ygt,n])
            return outproL

            #def compute_outer(l,Xgtv,Ygtv,n):
            #    n = T.cast(n,'float32')
            #    return Xgtv*Ygtv[l]/n
            #def compute_outer_L(i,L,Xgt,Ygt,n):
            #    Xgtv=Xgt[i,:]
            #    Ygtv=Ygt[i,:]
            #    outpro,updates = theano.scan(fn=compute_outer, outputs_info=None,sequences=T.arange(L),non_sequences=[Xgtv,Ygtv,n])
            #    return T.sum(outpro, axis=0)
            #outproL,updates = theano.scan(fn=compute_outer_L, outputs_info=None,sequences=T.arange(n),non_sequences=[L,Xgt,Ygt,n])
            #return T.sum(outproL, axis=0)

        barycenters,updates_3 = theano.scan(fn=compute_barycenter, outputs_info=None,sequences=T.arange(classes),non_sequences=[y_pred,yt,y_true,L])

        #print(barycenters)
        #print(barycenters[0][0].eval())
        #print(barycenters.shape.eval())

        def compute_prior(group,yt,y_true,N):
            Ygt = y_true[T.eq(yt, group).nonzero()[0], :]
            n = T.cast(Ygt.shape[0],'float32')
            #N = T.cast(N,'float32')
            return n/N

        prior,updates_4 = theano.scan(fn=compute_prior,outputs_info=None,sequences=T.arange(classes),non_sequences=[yt,y_true,T.cast(N,'float32')])
        #print(prior.eval())
        

        def compute_cov(group,barycenters,Xt,yt,y_true,L,N):
            Xgt = Xt[T.eq(yt, group).nonzero()[0], :]
            Ygt = y_true[T.eq(yt, group).nonzero()[0], :]
            n = Xgt.shape[0]
            barycenter = barycenters[group]
            #dim = Xgt.shape[1]
            def compute_out(l,Xgtv,Ygtv,barycenter,n):
                Bacv_com = barycenter[l]
                temp = T.reshape(Xgtv - Bacv_com, (1,Xgtv.shape[0]))
                #temp = T.cast(temp,)
                #temp = theano.shared(np.float32([[1,2,3,4]]))
                n = T.cast(n,'float32')
                #print(temp.eval())
                return Ygtv[l]*T.dot(temp.T,temp)/n
            def compute_out_L(i,L,Xgt,Ygt,barycenter,n):
                Xgtv=Xgt[i,:]
                Ygtv=Ygt[i,:]
                #Bacv = barycenters[i]
                out,updates_5 = theano.scan(fn=compute_out, outputs_info=None,sequences=T.arange(L),non_sequences=[Xgtv,Ygtv,barycenter,n])
                return T.sum(out,axis=0)
            outL,updates_6 = theano.scan(fn=compute_out_L, outputs_info=None,sequences=numpy_arange(n),non_sequences=[L,Xgt,Ygt,barycenter,n])
            n = T.cast(n,'float32')
            N = T.cast(N,'float32')
            prior = n/N
            return prior*T.sum(outL, axis=0)

        convs,updates_7 = theano.scan(fn=compute_cov, outputs_info=None,sequences=T.arange(classes),non_sequences=[barycenters,y_pred,yt,y_true,L,N])       
        #print(convs.eval())
        Sw_t = T.sum(convs,axis=0)
        #print(Sw_t.eval())

        #Sb_t = T.zeros(dim,dim)
        #for c_1 in classes:
        #   for c_2 in range(c_1+1,classes):
        #       for i in range(L):
        #           for j in range(L):
        #               temp = barycenter[c_1,i,:] - barycenter[c_2,j,:]
        #               Sb_t = Sb_t + prior[c_1]*prior[c_2]*Yinter[c_1,c_2,i,j]*T.dot(temp.T,temp)

        def compute_Sb(group,prior,barycenters,classes,Yinter,L):
            prior1 = prior[group]            
            Yinter1 = Yinter[group,:,:,:]
            def compute_Sb_per(c,group,prior,barycenters,classes,Yinter1,L):
                prior2 = prior[c]
                bac1 = barycenters[group]
                bac2 = barycenters[c]
                Yinter2 = Yinter1[c,:,:]
                def compute_Sb_in(l,bac1,bac2,Yinter2,L):
                    bac1v = bac1[l]
                    Yinter3 = Yinter2[l,:]
                    def compute_Sb_inner(l2,bac1v,bac2,Yiter3):
                        bac2v = bac2[l2]
                        temp = T.reshape(bac1v - bac2v, (1,bac1v.shape[0]))
                        return Yinter3[l2]*T.dot(temp.T,temp)
                    out3,updates_8 = theano.scan(fn=compute_Sb_inner, outputs_info=None,sequences=T.arange(L),non_sequences=[bac1v,bac2,Yinter3])
                    return T.sum(out3,axis=0)
                out2,updates_9 = theano.scan(fn=compute_Sb_in, outputs_info=None,sequences=T.arange(L),non_sequences=[bac1,bac2,Yinter2,L])
                return prior2*T.sum(out2,axis=0)
            out1,updates_10 = theano.scan(fn=compute_Sb_per, outputs_info=None,sequences=T.arange(group+1,classes),non_sequences=[group,prior,barycenters,classes,Yinter1,L])
            return prior1*T.sum(out1,axis=0)


        Sbs,updates = theano.scan(fn=compute_Sb, outputs_info=None,sequences=T.arange(classes-1),non_sequences=[prior,barycenters,classes,Yinter,L])
        Sb_t = T.sum(Sbs,axis=0)
        #print(Sb_t.eval())



        #def compute_cov(group, Xt, yt):
        #    Xgt = Xt[T.eq(yt, group).nonzero()[0], :]
        #    Xgt_bar = Xgt - T.mean(Xgt, axis=0)
        #    m = T.cast(Xgt_bar.shape[0], 'float32')
        #    return (1.0 / (m - 1)) * T.dot(Xgt_bar.T, Xgt_bar)

        # scan over groups
        #covs_t, updates = theano.scan(fn=compute_cov, outputs_info=None,
        #                              sequences=[groups], non_sequences=[y_pred, yt, y_true, L])

        # compute average covariance matrix (within scatter)
        #Sw_t = T.mean(covs_t, axis=0)

        # compute total scatter
        #Xt_bar = y_pred - T.mean(y_pred, axis=0)
        #m = T.cast(Xt_bar.shape[0], 'float32')
        #St_t = (1.0 / (m - 1)) * T.dot(Xt_bar.T, Xt_bar)

        # compute between scatter
        #Sb_t = St_t - Sw_t

        # cope for numerical instability (regularize)
        #print(Sw_t.shape.eval())
        #print(Sb_t.shape.eval())
        Sw_t += T.identity_like(Sw_t) * r
        #Sb_t += T.identity_like(Sb_t) * r

        # return T.cast(T.neq(yt[0], -1), 'float32')*T.nlinalg.trace(T.dot(T.nlinalg.matrix_inverse(St_t), Sb_t))

        # compute eigenvalues
        evals_t = T.slinalg.eigvalsh(Sb_t, Sw_t)

        # get eigenvalues
        top_k_evals = evals_t[-n_components:]

        # maximize variance between classes
        # (k smallest eigenvalues below threshold)
        thresh = T.min(top_k_evals) + margin
        top_k_evals = top_k_evals[(top_k_evals <= thresh).nonzero()]
        costs = T.mean(top_k_evals)
        #print(costs.eval())

        return -costs

    return inner_owda_objective



def main():
    data_size = 32
    input_dim = 6
    L = 3
    num_class = 4
    output_dim = 12
    # margin = 0.5
    x = np.random.rand(data_size, input_dim)
    w = np.random.rand(input_dim, output_dim)
    inputs = np.dot(x,w) #x.mm(w)
    # print(x.dtype)
    targets = np.random.rand(data_size, L)
    
    y_ = np.random.rand(data_size,1)
    #print(y_)
    for i in range(data_size):
        y_[i] = np.random.randint(0,num_class)

    targets_y = y_.astype(np.float32)  #Variable(torch.FloatTensor(y_))
    targets = np.concatenate((targets,targets_y),1)

    Yinter = np.random.rand(num_class,num_class,L,L)

    downdim = 3
    margin = 1


    myloss = owda_loss(n_components = downdim, margin=margin, Yinter=Yinter)
    a = myloss(targets, x)
    print(a)


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
