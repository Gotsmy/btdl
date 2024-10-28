using JuMP
using LinearAlgebra
using Ipopt
using DelimitedFiles
using TickTock
using JLD2
import DataFrames
import CSV
import ForwardDiff
import Dierckx

include("/home/users/mgotsmy/julia/dFBA/custom/postprocessing_02.jl")

function fed_batch_dFBA(N0,V0,nFE,S,t_max,t_min,V_max,c_G,glu,atp,oxy,xxx,pro,aon,dil,vlb,vub,d)
    A_idx, B_idx, C_idx, D_idx = get_idx_lists(vlb,vub)
    nA = size(A_idx,1)
    nB = size(B_idx,1)
    nC = size(C_idx,1)
    nD = size(D_idx,1)
    println("nA = ",nA)
    println("nB = ",nB)
    println("nC = ",nC)
    println("nD = ",nD)
    
    # number of reactions
    nR = size(S,2)
    # number of metabolites
    nM = size(S,1)
    # number of tracked metabolites 
    nN = length(N0)
    # number collocation points
    nCP = 3
    
    #--------------------------
    # SIMULATION HYPERPARAMETERS
    
    w    = 1e-6 # weigth for flux sum minimization on the inner objective
    phi1 = 1e+0 # factor on outer objective
    phi2 = 1e+1 # factor on inner objective
    phi3 = 1e-1

    #--------------------------
    # COLLOCATION AND RADAU PARAMETERS
    colmat = [0.19681547722366  -0.06553542585020 0.02377097434822;
              0.39442431473909  0.29207341166523 -0.04154875212600;
              0.37640306270047  0.51248582618842 0.11111111111111]
    radau  = [0.15505 0.64495 1.00000]

    #--------------------------
    # JuMP MODEL SETUP
    m = Model(optimizer_with_attributes(Ipopt.Optimizer, 
#            "warm_start_init_point" => "yes", 
            "print_level" => 4, 
            "linear_solver" => "ma27", 
            "max_iter" => Int(1e5),
	    "print_frequency_iter" => 100,
            "tol" => 1e-6, 
            "acceptable_iter" => 15, 
            "acceptable_tol" => 1e-3))

    #--------------------------
    # VARIABLE SET UP
    @variables(m, begin
            # Differential Equation Variables
            N[   1:nN, 1:nFE, 1:nCP]  # metabolite amounts [mmol]
            Ndot[1:nN, 1:nFE, 1:nCP]  # dN/dt [mmol/h]
            V[         1:nFE, 1:nCP]  # reactor volume [L]
            Vdot[      1:nFE, 1:nCP]  # dV/dt [L/h]
            q[   1:nR, 1:nFE       ]  # FBA fluxes [mmol/(g h)]

            # KKT Variables
            lambda_Sv[1:nM, 1:nFE]

            # A: both constrained
            alpha_ub_A[1:nA, 1:nFE]
            slack_ub_A[1:nA, 1:nFE]
            alpha_lb_A[1:nA, 1:nFE]
            slack_lb_A[1:nA, 1:nFE]

            # B: lb constrained, ub free
            alpha_lb_B[1:nB, 1:nFE]
            slack_lb_B[1:nB, 1:nFE]
            
            # C: lb free, ub constrained
            alpha_ub_C[1:nC, 1:nFE]
            slack_ub_C[1:nC, 1:nFE]

            # Process Variabels
            t_end       # total process length [h]
            rtFE[1:nFE] # relative length of finite elements [-]
            Fx[1:nFE]    # feed rate [L/h]
    end)

    #--------------------------
    # GUESSING START VALUES
    
    println("Setting start values.")
    for i in 1:nFE
      for j in 1:nCP
          for k in 1:nN
              set_start_value(N[k,i,j], N0[k])
          end
          set_start_value(V[i,j],V0)
      end
    end
    for i in 1:nFE
        set_start_value(rtFE[i], 1)
    end
    println("Finished setting start values.")

    #--------------------------
    # SET UP OBJECTIVE FUNCTION AND CONSTRAINTS
    
    # Uptake Rates as NLexpressions
    @NLexpressions(m, begin
        # glucose
        q_G[i=1:nFE], Fx[i]*c_G # L/h * mmol/L / g = mmol/(g h)
        # absolute values for length of finite elements (tFE)
        tFE[i=1:nFE], t_end/nFE*rtFE[i]
    end)
    
    @NLobjective(m, Max, +
          N[4,end,end]/t_end*1e+1 +
	  sum(N[4,i,end] for i in 1:nFE)/nFE*phi3 -
	    sum(
	    sum(- slack_lb_A[mc,i] for mc in 1:nA) +
	    sum(- slack_ub_A[mc,i] for mc in 1:nA) +
	    sum(- slack_lb_B[mc,i] for mc in 1:nB) +
	    sum(- slack_ub_C[mc,i] for mc in 1:nC)
	    for i in 1:nFE)/nFE)
    
    @NLconstraints(m, begin
        # NL process constraints
        NLc_q_G1[i=1:nFE], -q[glu,i] - q_G[i]                                 == 0 # qG -> feed rate
        NLc_q_G2[i=1:nFE],  q[glu,i]  +1.6832880759 +26.7371273713 * q[xxx,i] == 0 # µ  -> qG
        NLc_q_A1[i=1:nFE],  q[aon,i]  +5.0649596206 -36.6856368564 * q[xxx,i] >= 0 # µ  -> qA
        NLc_q_D1[i=1:nFE],  q[dil,i]  -1.2944612466 -10.3956639566 * q[xxx,i] == 0 # µ  -> qD

        NLc_t_end, sum(tFE[i] for i in 1:nFE) == t_end
            
        # NL control process constraints
#        NLc_t_switch,  sum(tFE[i] for i in 1:6) == 12.3
#        NLc_ctrl1[i=1:6],   q[xxx,i] - vub[xxx] == 0
#        NLc_ctrl2[i=7:nFE], q[xxx,i] - vlb[xxx] <= .001
        # NLc_ctrl3[i=9:nFE], q[aon,i] <= 0
            
        # NL KKT constraints, aka complementary slackness
        NLc_slack_lb_A[mc=1:nA,i=1:nFE], slack_lb_A[mc,i]  == (q[A_idx[mc],i] -vlb[A_idx[mc]])*alpha_lb_A[mc,i]/nA*phi2
        NLc_slack_ub_A[mc=1:nA,i=1:nFE], slack_ub_A[mc,i]  == (q[A_idx[mc],i] -vub[A_idx[mc]])*alpha_ub_A[mc,i]/nA*phi2
        NLc_slack_lb_B[mc=1:nB,i=1:nFE], slack_lb_B[mc,i]  == (q[B_idx[mc],i] -vlb[B_idx[mc]])*alpha_lb_B[mc,i]/nB*phi2
        NLc_slack_ub_C[mc=1:nC,i=1:nFE], slack_ub_C[mc,i]  == (q[C_idx[mc],i] -vub[C_idx[mc]])*alpha_ub_C[mc,i]/nC*phi2

        # INTEGRATION BY COLLOCATION
        # set up collocation equations - 2nd-to-nth point
        coll_N[l=1:nN, i=2:nFE, j=1:nCP], N[l,i,j] == N[l,i-1,nCP]+tFE[i]*sum(colmat[j,k]*Ndot[l,i,k] for k in 1:nCP)
        coll_V[        i=2:nFE, j=1:nCP], V[i,j]   == V[  i-1,nCP]+tFE[i]*sum(colmat[j,k]*Vdot[  i,k] for k in 1:nCP)
        # set up collocation equations - 1st point
        coll_N0[l in 1:nN, i=[1], j=1:nCP], N[l,i,j] == N0[l] + tFE[i]*sum(colmat[j,k]*Ndot[l,i,k] for k in 1:nCP)
        coll_V0[           i=[1], j=1:nCP], V[  i,j] == V0    + tFE[i]*sum(colmat[j,k]*Vdot[  i,k] for k in 1:nCP)
    end)

    #------------------------#
    # SET UP BOUNDS
    
    for mc in 1:nR
        for i in 1:nFE
            set_lower_bound(q[mc,i],vlb[mc])
            set_upper_bound(q[mc,i],vub[mc])
            for j in 1:nCP
                for n in 1:nN-1
                    set_lower_bound(N[n,i,j],0)
                end
                set_lower_bound(V[  i,j],V0)
                set_upper_bound(V[  i,j],V_max)
            end
        end
    end
    
    for i in 1:nFE
        set_lower_bound(rtFE[i],0.8)
        set_upper_bound(rtFE[i],1.2)
        set_lower_bound(Fx[i],0.)
        set_upper_bound(Fx[i],1.)
        for mc in 1:nA
            set_upper_bound(alpha_lb_A[mc,i],0)
            set_lower_bound(alpha_ub_A[mc,i],0)
        end
        for mc in 1:nB
            set_upper_bound(alpha_lb_B[mc,i],0)
        end
        for mc in 1:nC
            set_lower_bound(alpha_ub_C[mc,i],0)
        end
    end
    set_lower_bound(t_end,t_min)
    set_upper_bound(t_end,t_max)
    
    #------------------------#
    # SET UP OTHER CONSTRAINTS
    
    @constraints(m, begin
            #------------------------#
            # DIFFERENTIAL EQUATIONS

            # glucose
            m1[i=1:nFE, j=1:nCP], Ndot[1,i,j] == Fx[i]*c_G*N[2,i,j] + q[glu,i]*N[2,i,j] # mmol/h = L/h * mmol/L + mmol/(g h) * g
            # biomass
            m2[i=1:nFE, j=1:nCP], Ndot[2,i,j] == q[xxx,i]*N[2,i,j] # g/h = g/(g h) * g
            # sulfate
            m3[i=1:nFE, j=1:nCP], Ndot[3,i,j] == q[aon,i]*N[2,i,j]
            # product
            m4[i=1:nFE, j=1:nCP], Ndot[4,i,j] == q[pro,i]*N[2,i,j]
            # volume
            v1[i=1:nFE, j=1:nCP], Vdot[i,j]   == Fx[i]*N[2,i,j]

            #------------------------#
            # SYSTEM CONSTRAINTS

            c_S[mc=1:nM,i=1:nFE], sum(S[mc,k] * q[k,i] for k in 1:nR) == 0

            #------------------------#
            # KKT CONSTRAINTS

        # Lagr_A[mc=1:nA,i=1:nFE],  d[A_idx[mc]] + w*q[A_idx[mc],i]  + alpha_lb_A[mc,i] + alpha_ub_A[mc,i] +  sum(S[k,A_idx[mc]]*lambda_Sv[k,i] for k in 1:nM) == 0
        # Lagr_B[mc=1:nB,i=1:nFE],  d[B_idx[mc]] + w*q[B_idx[mc],i]  + alpha_lb_B[mc,i]                    +  sum(S[k,B_idx[mc]]*lambda_Sv[k,i] for k in 1:nM) == 0
        # Lagr_C[mc=1:nC,i=1:nFE],  d[C_idx[mc]] + w*q[C_idx[mc],i]                     + alpha_ub_C[mc,i] +  sum(S[k,C_idx[mc]]*lambda_Sv[k,i] for k in 1:nM) == 0
        # Lagr_D[mc=1:nD,i=1:nFE],  d[D_idx[mc]] + w*q[D_idx[mc],i]                                        +  sum(S[k,D_idx[mc]]*lambda_Sv[k,i] for k in 1:nM) == 0
        # non-parsimonious    
        Lagr_A[mc=1:nA,i=1:nFE],  d[A_idx[mc]]  + alpha_lb_A[mc,i] + alpha_ub_A[mc,i] +  sum(S[k,A_idx[mc]]*lambda_Sv[k,i] for k in 1:nM) == 0
        Lagr_B[mc=1:nB,i=1:nFE],  d[B_idx[mc]]  + alpha_lb_B[mc,i]                    +  sum(S[k,B_idx[mc]]*lambda_Sv[k,i] for k in 1:nM) == 0
        Lagr_C[mc=1:nC,i=1:nFE],  d[C_idx[mc]]                     + alpha_ub_C[mc,i] +  sum(S[k,C_idx[mc]]*lambda_Sv[k,i] for k in 1:nM) == 0
        Lagr_D[mc=1:nD,i=1:nFE],  d[D_idx[mc]]                                        +  sum(S[k,D_idx[mc]]*lambda_Sv[k,i] for k in 1:nM) == 0

        c_alpha_lb_A[mc=1:nA,i=1:nFE],   alpha_lb_A[mc,i]    <= 0
        c_alpha_ub_A[mc=1:nA,i=1:nFE],   alpha_ub_A[mc,i]    >= 0
        c_alpha_lb_B[mc=1:nB,i=1:nFE],   alpha_lb_B[mc,i]    <= 0
        c_alpha_ub_C[mc=1:nC,i=1:nFE],   alpha_ub_C[mc,i]    >= 0    
    end)

    #------------------------#
    # MODEL OPTIMIZATION
    
    println("Model Preprocessing Finished.")
    tick()
    solveNLP = JuMP.optimize!
    status = solveNLP(m)
    t = tok()
    #tock()
    println("Model Finished Optimization")

    # Read out model parameters
    N_    = JuMP.value.(N[:,:,:])
    Ndot_ = JuMP.value.(Ndot[:,:,:])
    V_    = JuMP.value.(V[:,:])
    Vdot_ = JuMP.value.(Vdot[:,:])
    q_    = JuMP.value.(q[:,:])
    lambda_Sv_ = JuMP.value.(lambda_Sv[:,:])
    alpha_ub_A_ = JuMP.value.(alpha_ub_A[:,:])
    slack_ub_A_ = JuMP.value.(slack_ub_A[:,:])
    alpha_lb_A_ = JuMP.value.(alpha_lb_A[:,:])
    slack_lb_A_ = JuMP.value.(slack_lb_A[:,:])
    alpha_lb_B_ = JuMP.value.(alpha_lb_B[:,:])
    slack_lb_B_ = JuMP.value.(slack_lb_B[:,:])
    alpha_ub_C_ = JuMP.value.(alpha_ub_C[:,:])
    slack_ub_C_ = JuMP.value.(slack_ub_C[:,:])
    tFE_   = JuMP.value.(tFE[:])
    Fx_     = JuMP.value.(Fx[:])
    # S0_    = JuMP.value.(S0)
    rtFE_  = JuMP.value.(rtFE[:])
    t_end_ = JuMP.value.(t_end)
    
    @JLD2.save dirname*"/variables.jld2" N_ Ndot_ V_ Vdot_ q_ lambda_Sv_ tFE_ Fx_ rtFE_ t_end_ alpha_ub_A_ slack_ub_A_ alpha_lb_A_ slack_lb_A_ alpha_lb_B_ slack_lb_B_ alpha_ub_C_ slack_ub_C_ A_idx B_idx C_idx D_idx

    sum_slack = 0.0*Vector{Float64}(undef,nFE)
    for i in 1:nFE
        sum_slack[i] = sum(slack_lb_A_[j,i] for j in 1:nA; init=0) + 
                       sum(slack_ub_A_[j,i] for j in 1:nA; init=0) + 
                       sum(slack_lb_B_[j,i] for j in 1:nB; init=0) +
                       sum(slack_ub_C_[j,i] for j in 1:nC; init=0)
    end

return  N_, Ndot_, V_, Vdot_, q_, tFE_, sum_slack, m, t
end

function run_kkt_simulation()

    # load stoichiometric matrix & flux bounds
    S   = readdlm("/home/users/mgotsmy/julia/240426_extended_boku_model/analysis/btdl_pathway_S.csv")
    vlb = readdlm("/home/users/mgotsmy/julia/240426_extended_boku_model/analysis/btdl_pathway_LB.csv")[:,1]
    vub = readdlm("/home/users/mgotsmy/julia/240426_extended_boku_model/analysis/btdl_pathway_UB.csv")[:,1]
    println("nR = ",size(vlb,1)," = ",size(vub,1))
    
    # reaction indices
    glu = 49  # glucose uptake
    atp = 14  # atp maintenance
    xxx = 22  # biomass growth
    pro = 91  # butane23diol production
    oxy = 57  # oxygen uptake
    aon = 95  # acetoin exchange
    dil = 92  # diol production
    psk = 94  # butane23diol sink
    vlb[xxx] = 0.17099345812364167
    
    
    
    # initial bioprocess state variables
    V0 =  0.781 # L
    G0 =  0.0 # mmol
    X0 =  0.71 # g
    S0 =  0 # mmol
    N0 = [G0,X0,S0,0]
    
    # glucose concentration in feed medium
    c_G = 800/0.18015588 # g/L /(g/mmol) = mmol/L, i.el max solubility of glucose in water

    # process length and state variable bounds
    t_max = 31.0 # h
    t_min = 31.0  # h
    V_max = 1  # L
    nFE   = 30

    # KKT FBA objective function: optimize product
    nR = size(S,2)
    d  = 0.0*Vector{Float64}(undef,nR) 
    d[psk] = -1

    N_, Ndot_, V_, Vdot_, q_, tFE_, sum_slack, m_, t_ = fed_batch_dFBA(N0,V0,nFE,S,t_max,t_min,V_max,c_G,glu,atp,oxy,xxx,pro,aon,dil,vlb,vub,d)
    println("DONE")
          
    #-------------------------
    # SAVING ALL INTERESTING FILES
    
    summary = "#-------------------------\n" * 
              "Termination Status\n" * string(termination_status(m_)) * 
              "\n\nTime Elapsed\n" * string(t_) *
              "\n\nObjective Value\n" * string(JuMP.objective_value(m_)) * 
              "\n\nNormalized Complementary Slackness\n" * string(-(sum(-sum_slack))/nFE) * 
              "\n\nFinite Elements\n" * string([round(i,digits=2) for i in tFE_]) * 
              "\n\nComplementary Slackness per Finite Element\n" * string([round(sum_slack[i],digits=2) for i in 1:nFE]) * 
              "\n\nFeed Rates\n" * string([round(i,digits=3) for i in Vdot_[:,1]]) *
              "\n\nGrowth Rates [h-1]\n" * string([round(i,digits=3) for i in q_[xxx,:]]) * 
              "\n\nFinal Product Amount [mmol]\n" * string(N_[4,end,end]) * 
              "\n\nFinal Biomass Amount [g]\n" * string(N_[2,end,end]) * 
              "\n\nProcess End\n" * string(round(sum(tFE_),digits=2)) * 
              "\n#-------------------------"
    println(summary)
    
    #-------------------------
    # POST PROCESSING

    df = DataFrames.DataFrame(hcat(
            get_time_points(tFE_),
            get_points(V_,tFE_,V0),
            get_points(N_,tFE_,[N0[1],N0[2],N0[3],N0[4]]),
            get_points(Vdot_,tFE_),
            get_points(Ndot_,tFE_),
            get_fluxes(q_,[glu,xxx,aon,pro,dil,atp,oxy],tFE_)),
        ["t","V","G","X","S","P","r_V","r_G","r_X","r_S","r_P","q_G","q_X","q_S","q_P","q_D","q_M","q_O"]);
    
    write(   dirname*"/summary.txt",summary)
    writedlm(dirname*"/q.csv",  q_, ',')
    writedlm(dirname*"/vlb.csv",vlb,',')
    writedlm(dirname*"/vub.csv",vub,',')
    CSV.write(dirname*"/df.csv",df) 
end

#-------------------------
# SCRIPT START

println("Start Script")
scriptname = PROGRAM_FILE
dirname = scriptname[begin:end-3]
println("Creating results directory: ",dirname)
mkpath(dirname)

#-------------------------
# RUN MODEL

debug = true
if debug == true
    run_kkt_simulation()
else
    # sometimes errors happen during initialization, they usually desappear when rerunning the simulation
    # thus in debug = false mode, these errors are escapted.
    global worked = false
    global nTRY = 0
    while worked == false
        try
            run_kkt_simulation()
            global worked = true
        catch e
            if isa(e,ErrorException)
                global nTRY += 1
                println("Error During Initialization. Retrying ... (",nTRY,")")
            else
                rethrow(e)
            end
        end
    end
end

println("Script Ended")
