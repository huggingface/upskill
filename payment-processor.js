class PaymentProcessor {
  constructor(apiClient) {
    this.apiClient = apiClient;
  }

  async processPayment(amount, currency, paymentMethod) {
    if (!amount || amount <= 0) {
      throw new Error('Invalid amount');
    }

    if (!currency || currency.length !== 3) {
      throw new Error('Invalid currency code');
    }

    if (!paymentMethod) {
      throw new Error('Payment method required');
    }

    try {
      const response = await this.apiClient.post('/payments', {
        amount,
        currency,
        paymentMethod
      });

      return {
        success: true,
        transactionId: response.data.id,
        status: response.data.status
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async refundPayment(transactionId, refundAmount) {
    if (!transactionId) {
      throw new Error('Transaction ID required');
    }

    if (!refundAmount || refundAmount <= 0) {
      throw new Error('Invalid refund amount');
    }

    try {
      const response = await this.apiClient.post(`/payments/${transactionId}/refund`, {
        amount: refundAmount
      });

      return {
        success: true,
        refundId: response.data.id,
        status: response.data.status
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async getPaymentStatus(transactionId) {
    if (!transactionId) {
      throw new Error('Transaction ID required');
    }

    try {
      const response = await this.apiClient.get(`/payments/${transactionId}`);
      return {
        status: response.data.status,
        amount: response.data.amount,
        currency: response.data.currency
      };
    } catch (error) {
      throw new Error(`Failed to get payment status: ${error.message}`);
    }
  }
}

module.exports = PaymentProcessor;
